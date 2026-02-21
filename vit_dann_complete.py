#!/usr/bin/env python3
"""
ViT-B/16 + DANN Training Pipeline with Complete Data Logging
Saves all raw data needed to regenerate any graph later.

Meta-spec requirements:
- Dataset: CS4/DATASET/{LC25000, NCT_CRC_HE_100K}/{cancerous, non_cancerous}
- Stratified splits: train=12,000, val=1,200, test=600 by (domain × label)
- Model: ViT-B/16 (augreg_in21k) + DANN
- Training: 3 phases over 30 epochs (frozen→DANN→unfrozen)
- Seeds: 0, 1, 2 (deterministic)
- Thresholds: T=0.50 and T=Youden(val)
- All figures at 300 DPI
- Resume-safe with per-epoch checkpoints
"""

import os
import sys
import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as T
from torchvision.datasets.folder import default_loader

import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Global configuration matching the spec exactly."""
    
    # Paths
    DATA_ROOT = Path("/content/drive/MyDrive/CS4/DATASET")  # Change as needed
    OUTPUT_ROOT = Path("runs")
    
    # Dataset splits (exact sizes)
    TRAIN_SIZE = 12000
    VAL_SIZE = 1200
    TEST_SIZE = 600
    
    # Training phases
    TOTAL_EPOCHS = 30
    PHASE1_EPOCHS = 10  # Frozen, cancer head only
    PHASE2_EPOCHS = 20  # Frozen, cancer + domain (DANN)
    PHASE3_EPOCHS = 30  # Unfrozen, full DANN
    
    # Model
    MODEL_NAME = "vit_base_patch16_224.augreg_in21k"
    IMG_SIZE = 224
    NUM_CLASSES = 2
    NUM_DOMAINS = 2
    
    # Training hyperparameters
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    
    # Learning rates by phase
    LR_PHASE1 = 1e-3  # heads only
    LR_PHASE2 = 1e-3  # heads only
    LR_PHASE3_ENCODER = 1e-5  # encoder
    LR_PHASE3_HEADS = 1e-4  # heads
    
    # DANN
    LAMBDA_DANN = 0.3
    
    # Thresholds
    THRESHOLD_DEFAULT = 0.50
    
    # Seeds
    SEEDS = [0, 1, 2]
    
    # Augmentation
    AUGMENT_SCALE = (0.8, 1.0)
    AUGMENT_JITTER = 0.2
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Figure settings
    DPI = 300
    
    # Domain and label mapping
    LABEL_MAP = {"non_cancerous": 0, "cancerous": 1}
    DOMAIN_MAP = {"LC25000": 0, "NCT-CRC-HE-100K": 1}
    
    # Reverse mappings
    LABEL_NAMES = {0: "normal", 1: "cancer"}
    DOMAIN_NAMES = {0: "LC25000", 1: "NCT-CRC-HE-100K"}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def setup_logger(log_dir: Path, name: str = "training") -> logging.Logger:
    """Setup logger that writes to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_dir / "console.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_environment_info(output_path: Path):
    """Save environment information."""
    import platform
    import torch
    import torchvision
    import timm
    
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    return info


# ============================================================================
# DATA PREPARATION
# ============================================================================

def scan_dataset(data_root: Path) -> pd.DataFrame:
    """
    Scan the dataset directory and create master dataframe.
    
    Expected structure:
    CS4/DATASET/
    ├── LC25000/
    │   ├── cancerous/
    │   └── non_cancerous/
    └── NCT_CRC_HE_100K/
        ├── cancerous/
        └── non_cancerous/
    
    Returns:
        DataFrame with columns: path, label, domain
    """
    records = []
    
    for domain_dir in data_root.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain_name = domain_dir.name
        
        for label_dir in domain_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            label_name = label_dir.name
            
            if label_name not in Config.LABEL_MAP:
                continue
            
            # Scan all images
            for img_path in label_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    records.append({
                        'path': str(img_path),
                        'label': Config.LABEL_MAP[label_name],
                        'domain': domain_name
                    })
    
    df = pd.DataFrame(records)
    
    # Map domain names to indices
    domain_to_idx = {}
    for domain in df['domain'].unique():
        if domain in Config.DOMAIN_MAP:
            domain_to_idx[domain] = Config.DOMAIN_MAP[domain]
        else:
            # Auto-assign if not in map
            domain_to_idx[domain] = len(domain_to_idx)
    
    df['domain_idx'] = df['domain'].map(domain_to_idx)
    
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Stratify by (domain, label) to ensure balanced representation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stratification column
    df['strat_key'] = df['domain'] + '_' + df['label'].astype(str)
    
    # Check if we have enough data
    total_needed = Config.TRAIN_SIZE + Config.VAL_SIZE + Config.TEST_SIZE
    if len(df) < total_needed:
        raise ValueError(f"Not enough images! Have {len(df)}, need {total_needed}. "
                        f"Please ensure you have at least {total_needed} images in your dataset.")
    
    # Calculate test size as fraction to ensure we get close to 600
    # We'll use test_size parameter and then verify
    test_fraction = Config.TEST_SIZE / len(df)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df['strat_key'],
        random_state=random_state
    )
    
    # If test size is not exactly 600, adjust by taking exactly 600 samples
    if len(test_df) != Config.TEST_SIZE:
        # Sample exactly 600 from test_df, stratified
        test_df = test_df.groupby('strat_key', group_keys=False).apply(
            lambda x: x.sample(n=int(np.ceil(Config.TEST_SIZE * len(x) / len(test_df))), 
                              random_state=random_state)
        ).iloc[:Config.TEST_SIZE]
        
        # Add back the remaining samples to train_val
        all_test_paths = set(test_df['path'])
        remaining = df[~df['path'].isin(all_test_paths)]
        train_val_df = remaining
    
    # Calculate val size as fraction of train_val
    val_fraction = Config.VAL_SIZE / len(train_val_df)
    
    # Second split: separate train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df['strat_key'],
        random_state=random_state
    )
    
    # If val size is not exactly 1200, adjust
    if len(val_df) != Config.VAL_SIZE:
        val_df = val_df.groupby('strat_key', group_keys=False).apply(
            lambda x: x.sample(n=int(np.ceil(Config.VAL_SIZE * len(x) / len(val_df))), 
                              random_state=random_state)
        ).iloc[:Config.VAL_SIZE]
        
        # Add back remaining to train
        all_val_paths = set(val_df['path'])
        remaining_train = train_val_df[~train_val_df['path'].isin(all_val_paths)]
        train_df = remaining_train
    
    # Final adjustment for train to get exactly 12000
    if len(train_df) != Config.TRAIN_SIZE:
        if len(train_df) > Config.TRAIN_SIZE:
            # Sample down
            train_df = train_df.groupby('strat_key', group_keys=False).apply(
                lambda x: x.sample(n=int(np.ceil(Config.TRAIN_SIZE * len(x) / len(train_df))), 
                                  random_state=random_state)
            ).iloc[:Config.TRAIN_SIZE]
        else:
            # We have less than needed, this shouldn't happen but handle it
            raise ValueError(f"Not enough samples for training set: {len(train_df)} < {Config.TRAIN_SIZE}")
    
    # Verify sizes
    assert len(train_df) == Config.TRAIN_SIZE, f"Train size mismatch: {len(train_df)} != {Config.TRAIN_SIZE}"
    assert len(val_df) == Config.VAL_SIZE, f"Val size mismatch: {len(val_df)} != {Config.VAL_SIZE}"
    assert len(test_df) == Config.TEST_SIZE, f"Test size mismatch: {len(test_df)} != {Config.TEST_SIZE}"
    
    # Verify no overlap
    train_paths = set(train_df['path'])
    val_paths = set(val_df['path'])
    test_paths = set(test_df['path'])
    
    assert len(train_paths & val_paths) == 0, "Train/Val overlap detected!"
    assert len(train_paths & test_paths) == 0, "Train/Test overlap detected!"
    assert len(val_paths & test_paths) == 0, "Val/Test overlap detected!"
    
    # Save splits
    train_df.to_csv(output_dir / "train_12k.csv", index=False)
    val_df.to_csv(output_dir / "val_1200.csv", index=False)
    test_df.to_csv(output_dir / "test_600.csv", index=False)
    
    # Create manifest (convert tuple keys to strings for JSON)
    train_counts = {f"{k[0]}_{k[1]}": v for k, v in train_df.groupby(['domain', 'label']).size().to_dict().items()}
    val_counts = {f"{k[0]}_{k[1]}": v for k, v in val_df.groupby(['domain', 'label']).size().to_dict().items()}
    test_counts = {f"{k[0]}_{k[1]}": v for k, v in test_df.groupby(['domain', 'label']).size().to_dict().items()}
    
    manifest = {
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
        "total_train": len(train_df),
        "total_val": len(val_df),
        "total_test": len(test_df)
    }
    
    with open(output_dir / "split_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Compute checksums
    with open(output_dir / "checksums.txt", 'w') as f:
        for csv_file in ["train_12k.csv", "val_1200.csv", "test_600.csv"]:
            checksum = compute_sha256(output_dir / csv_file)
            f.write(f"{checksum}  {csv_file}\n")
    
    return train_df, val_df, test_df


# ============================================================================
# DATASET CLASS
# ============================================================================

class CancerDataset(Dataset):
    """Dataset for cancer image classification."""
    
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Verify all paths exist
        for idx, row in self.df.iterrows():
            if not Path(row['path']).exists():
                raise FileNotFoundError(f"Image not found: {row['path']}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img = default_loader(row['path'])
        
        if self.transform:
            img = self.transform(img)
        
        label = int(row['label'])
        domain = int(row['domain_idx'])
        
        return img, label, domain, idx


def get_transforms():
    """Get train and eval transforms."""
    
    train_transform = T.Compose([
        T.RandomResizedCrop(Config.IMG_SIZE, scale=Config.AUGMENT_SCALE),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(
            brightness=Config.AUGMENT_JITTER,
            contrast=Config.AUGMENT_JITTER
        ),
        T.ToTensor(),
        T.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])
    
    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(Config.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])
    
    return train_transform, eval_transform


def create_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Create weighted sampler to balance (domain, label) combinations.
    """
    # Create combined key
    df['combo'] = df['domain_idx'].astype(str) + '_' + df['label'].astype(str)
    
    # Count samples per combo
    combo_counts = df['combo'].value_counts()
    
    # Compute weights (inverse frequency)
    weights = df['combo'].map(lambda x: 1.0 / combo_counts[x])
    
    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(df),
        replacement=True
    )
    
    return sampler


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class ViTDANN(nn.Module):
    """ViT-B/16 with DANN for domain adaptation."""
    
    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        num_classes: int = Config.NUM_CLASSES,
        num_domains: int = Config.NUM_DOMAINS,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1]
        
        # Cancer classification head
        self.cancer_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Domain classification head (with GRL)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_domains)
        )
    
    def forward(self, x, lambda_=1.0, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images
            lambda_: DANN gradient reversal strength
            return_features: If True, also return backbone features
        
        Returns:
            cancer_logits, domain_logits, (features if requested)
        """
        # Extract features
        features = self.backbone(x)
        
        # Cancer classification
        cancer_logits = self.cancer_head(features)
        
        # Domain classification with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, lambda_)
        domain_logits = self.domain_head(reversed_features)
        
        if return_features:
            return cancer_logits, domain_logits, features
        
        return cancer_logits, domain_logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class Trainer:
    """Main training class."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        seed: int,
        logger: logging.Logger
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.seed = seed
        self.logger = logger
        
        # Create subdirectories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.log_dir = output_dir / "logs"
        self.figure_dir = output_dir / "figures"
        self.prediction_dir = output_dir / "predictions"
        self.metric_dir = output_dir / "metrics"
        self.deploy_dir = output_dir / "deploy"
        self.raw_data_dir = output_dir / "raw_data"  # NEW: for graph regeneration
        
        for dir in [self.checkpoint_dir, self.log_dir, self.figure_dir,
                    self.prediction_dir, self.metric_dir, self.deploy_dir,
                    self.raw_data_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.current_epoch = 0
        self.history = []
        
        # Optimizers and schedulers (will be initialized per phase)
        self.optimizer = None
        self.scheduler = None
        
        # Loss functions
        self.cancer_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint with all state."""
        checkpoint = {
            'epoch': epoch,
            'seed': self.seed,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'rng_state': {
                'python': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            'history': self.history,
            'config': {
                'model_name': Config.MODEL_NAME,
                'img_size': Config.IMG_SIZE,
                'batch_size': Config.BATCH_SIZE,
                'num_classes': Config.NUM_CLASSES,
                'num_domains': Config.NUM_DOMAINS
            }
        }
        
        # Save epoch checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:02d}.pth")
        
        # Save best if applicable
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_val_auc.pth")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and restore state."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore RNG states
        np.random.set_state(checkpoint['rng_state']['python'])
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        if torch.cuda.is_available() and checkpoint['rng_state']['cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
        
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def setup_phase(self, phase: int):
        """Setup optimizer and scheduler for each phase."""
        self.logger.info(f"Setting up Phase {phase}")
        
        if phase == 1:
            # Phase 1: Freeze backbone, train cancer head only
            self.model.freeze_backbone()
            params = list(self.model.cancer_head.parameters())
            self.optimizer = Adam(params, lr=Config.LR_PHASE1)
            self.logger.info("Phase 1: Backbone frozen, cancer head only, LR=1e-3")
            
        elif phase == 2:
            # Phase 2: Freeze backbone, train cancer + domain heads
            self.model.freeze_backbone()
            params = (
                list(self.model.cancer_head.parameters()) +
                list(self.model.domain_head.parameters())
            )
            self.optimizer = Adam(params, lr=Config.LR_PHASE2)
            self.logger.info("Phase 2: Backbone frozen, cancer+domain heads, LR=1e-3, λ=0.3")
            
        elif phase == 3:
            # Phase 3: Unfreeze backbone, different LRs
            self.model.unfreeze_backbone()
            params = [
                {'params': self.model.backbone.parameters(), 'lr': Config.LR_PHASE3_ENCODER},
                {'params': self.model.cancer_head.parameters(), 'lr': Config.LR_PHASE3_HEADS},
                {'params': self.model.domain_head.parameters(), 'lr': Config.LR_PHASE3_HEADS}
            ]
            self.optimizer = Adam(params)
            self.logger.info("Phase 3: Backbone unfrozen, LR(encoder)=1e-5, LR(heads)=1e-4, λ=0.3")
        
        # Scheduler (cosine annealing for remaining epochs)
        epochs_in_phase = {
            1: Config.PHASE1_EPOCHS,
            2: Config.PHASE2_EPOCHS - Config.PHASE1_EPOCHS,
            3: Config.PHASE3_EPOCHS - Config.PHASE2_EPOCHS
        }[phase]
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs_in_phase)
    
    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        cancer_loss_sum = 0.0
        domain_loss_sum = 0.0
        correct = 0
        total = 0
        
        # Determine if we use DANN
        use_dann = (phase >= 2)
        lambda_dann = Config.LAMBDA_DANN if use_dann else 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{Config.TOTAL_EPOCHS} [Train]")
        
        for images, labels, domains, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            cancer_logits, domain_logits = self.model(images, lambda_=lambda_dann)
            
            # Cancer loss
            cancer_loss = self.cancer_criterion(cancer_logits, labels)
            
            # Domain loss (only if DANN active)
            if use_dann:
                domain_loss = self.domain_criterion(domain_logits, domains)
                loss = cancer_loss + domain_loss
                domain_loss_sum += domain_loss.item()
            else:
                loss = cancer_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            cancer_loss_sum += cancer_loss.item()
            
            _, predicted = cancer_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_cancer_loss': cancer_loss_sum / len(self.train_loader),
            'train_acc': 100. * correct / total
        }
        
        if use_dann:
            metrics['train_domain_loss'] = domain_loss_sum / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        
        all_labels = []
        all_probs = []
        all_domains = []
        all_indices = []
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Evaluating [{split}]")
        
        for images, labels, domains, indices in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (no gradient reversal needed for eval)
            cancer_logits, _ = self.model(images, lambda_=0.0)
            
            # Loss
            loss = self.cancer_criterion(cancer_logits, labels)
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = cancer_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions
            probs = F.softmax(cancer_logits, dim=1)[:, 1].cpu().numpy()  # Prob of cancer class
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_domains.extend(domains.cpu().numpy())
            all_indices.extend(indices.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_domains = np.array(all_domains)
        
        # Compute metrics
        auroc = roc_auc_score(all_labels, all_probs)
        
        metrics = {
            f'{split}_loss': total_loss / len(loader),
            f'{split}_acc': 100. * correct / total,
            f'{split}_auroc': auroc
        }
        
        # Save predictions if requested
        predictions = {
            'indices': all_indices,
            'labels': all_labels,
            'probs': all_probs,
            'domains': all_domains
        }
        
        return metrics, predictions
    
    def train(self, resume: bool = False):
        """Main training loop."""
        
        # Resume if requested
        if resume:
            checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pth"))
            if checkpoints:
                latest = checkpoints[-1]
                self.load_checkpoint(latest)
        
        # Determine starting phase and epoch
        start_epoch = self.current_epoch + 1 if resume else 1
        
        best_val_auc = 0.0
        
        for epoch in range(start_epoch, Config.TOTAL_EPOCHS + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPOCH {epoch}/{Config.TOTAL_EPOCHS}")
            self.logger.info(f"{'='*60}")
            
            # Determine phase
            if epoch <= Config.PHASE1_EPOCHS:
                phase = 1
            elif epoch <= Config.PHASE2_EPOCHS:
                phase = 2
            else:
                phase = 3
            
            # Setup phase (if first epoch of phase)
            if epoch in [1, Config.PHASE1_EPOCHS + 1, Config.PHASE2_EPOCHS + 1]:
                self.setup_phase(phase)
            
            # Train
            train_metrics = self.train_epoch(epoch, phase)
            
            # Validate
            val_metrics, val_predictions = self.evaluate(self.val_loader, split="val")
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'phase': phase}
            self.history.append(epoch_metrics)
            
            # Log
            self.logger.info(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                           f"Train Acc: {train_metrics['train_acc']:.2f}%")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                           f"Val Acc: {val_metrics['val_acc']:.2f}% | "
                           f"Val AUROC: {val_metrics['val_auroc']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_auroc'] > best_val_auc
            if is_best:
                best_val_auc = val_metrics['val_auroc']
                self.logger.info(f"New best val AUROC: {best_val_auc:.4f}")
            
            self.current_epoch = epoch
            self.save_checkpoint(epoch, is_best=is_best)
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.log_dir / "training_history.csv", index=False)
        
        # Save training config
        config_dict = {
            'model_name': Config.MODEL_NAME,
            'img_size': Config.IMG_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'num_classes': Config.NUM_CLASSES,
            'num_domains': Config.NUM_DOMAINS,
            'total_epochs': Config.TOTAL_EPOCHS,
            'phase1_epochs': Config.PHASE1_EPOCHS,
            'phase2_epochs': Config.PHASE2_EPOCHS,
            'phase3_epochs': Config.PHASE3_EPOCHS,
            'lr_phase1': Config.LR_PHASE1,
            'lr_phase2': Config.LR_PHASE2,
            'lr_phase3_encoder': Config.LR_PHASE3_ENCODER,
            'lr_phase3_heads': Config.LR_PHASE3_HEADS,
            'lambda_dann': Config.LAMBDA_DANN,
            'seed': self.seed,
            'train_size': Config.TRAIN_SIZE,
            'val_size': Config.VAL_SIZE,
            'test_size': Config.TEST_SIZE,
            'augmentation': {
                'scale': Config.AUGMENT_SCALE,
                'jitter': Config.AUGMENT_JITTER
            },
            'normalization': {
                'mean': Config.IMAGENET_MEAN,
                'std': Config.IMAGENET_STD
            },
            'label_map': Config.LABEL_MAP,
            'domain_map': Config.DOMAIN_MAP
        }
        
        with open(self.log_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info("\nTraining complete!")
        self.logger.info(f"Best validation AUROC: {best_val_auc:.4f}")


# ============================================================================
# EVALUATION & METRICS (SAVES RAW DATA FOR GRAPHS)
# ============================================================================

def compute_youden_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict]:
    """Compute Youden's J statistic optimal threshold."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Youden's J = sensitivity + specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    info = {
        'threshold': float(best_threshold),
        'sensitivity': float(tpr[best_idx]),
        'specificity': float(1 - fpr[best_idx]),
        'j_score': float(j_scores[best_idx])
    }
    
    return best_threshold, info


def compute_all_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Compute comprehensive metrics at a given threshold."""
    
    preds = (probs >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision': precision_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auroc': roc_auc_score(labels, probs),
        'auprc': average_precision_score(labels, probs),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'threshold': float(threshold)
    }
    
    return metrics


def compute_per_domain_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    domains: np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """Compute metrics separately for each domain."""
    
    results = []
    
    for domain_idx in np.unique(domains):
        mask = (domains == domain_idx)
        domain_labels = labels[mask]
        domain_probs = probs[mask]
        
        metrics = compute_all_metrics(domain_labels, domain_probs, threshold)
        metrics['domain'] = Config.DOMAIN_NAMES.get(domain_idx, f"domain_{domain_idx}")
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def save_raw_curve_data(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    curve_type: str = "roc"
):
    """
    Save raw ROC or PR curve data for future graph regeneration.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        output_path: Where to save the data
        curve_type: "roc" or "pr"
    """
    if curve_type == "roc":
        fpr, tpr, thresholds = roc_curve(labels, probs)
        df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'threshold': thresholds
        })
    elif curve_type == "pr":
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        # Note: precision_recall_curve returns n+1 elements for precision/recall
        df = pd.DataFrame({
            'precision': precision[:-1],
            'recall': recall[:-1],
            'threshold': thresholds
        })
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")
    
    df.to_csv(output_path, index=False)


def evaluate_and_save(
    trainer: Trainer,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logger: logging.Logger
):
    """
    Complete evaluation pipeline:
    1. Get predictions on val and test
    2. Compute Youden threshold on val
    3. Evaluate test at both T=0.50 and T=Youden
    4. Save all metrics and figures
    5. SAVE RAW DATA for graph regeneration
    """
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION PHASE")
    logger.info("="*60)
    
    # ========================================================================
    # 1. Get predictions
    # ========================================================================
    
    logger.info("\nGenerating predictions...")
    
    # Load best model
    best_checkpoint = trainer.checkpoint_dir / "best_val_auc.pth"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
        logger.info("Loaded best model from validation AUROC")
    
    # Validation predictions
    _, val_preds_dict = trainer.evaluate(trainer.val_loader, split="val")
    val_labels = val_preds_dict['labels']
    val_probs = val_preds_dict['probs']
    val_domains = val_preds_dict['domains']
    val_indices = val_preds_dict['indices']
    
    # Test predictions
    _, test_preds_dict = trainer.evaluate(trainer.test_loader, split="test")
    test_labels = test_preds_dict['labels']
    test_probs = test_preds_dict['probs']
    test_domains = test_preds_dict['domains']
    test_indices = test_preds_dict['indices']
    
    # Save predictions CSVs
    val_pred_df = pd.DataFrame({
        'image_id': [val_df.iloc[i]['path'] for i in val_indices],
        'y_true': val_labels,
        'y_prob': val_probs,
        'domain': [Config.DOMAIN_NAMES[d] for d in val_domains]
    })
    val_pred_df.to_csv(trainer.prediction_dir / "val_predictions.csv", index=False)
    
    test_pred_df = pd.DataFrame({
        'image_id': [test_df.iloc[i]['path'] for i in test_indices],
        'y_true': test_labels,
        'y_prob': test_probs,
        'domain': [Config.DOMAIN_NAMES[d] for d in test_domains]
    })
    test_pred_df.to_csv(trainer.prediction_dir / "test_predictions.csv", index=False)
    
    logger.info(f"Saved predictions to {trainer.prediction_dir}")
    
    # ========================================================================
    # 2. Compute Youden threshold
    # ========================================================================
    
    logger.info("\nComputing Youden threshold on validation set...")
    
    youden_threshold, youden_info = compute_youden_threshold(val_labels, val_probs)
    
    with open(trainer.metric_dir / "youden_info.json", 'w') as f:
        json.dump(youden_info, f, indent=2)
    
    logger.info(f"Youden threshold: {youden_threshold:.4f}")
    logger.info(f"  Sensitivity: {youden_info['sensitivity']:.4f}")
    logger.info(f"  Specificity: {youden_info['specificity']:.4f}")
    
    # ========================================================================
    # 3. Evaluate test set at both thresholds
    # ========================================================================
    
    logger.info("\nEvaluating test set...")
    
    # T = 0.50
    metrics_t050 = compute_all_metrics(test_labels, test_probs, Config.THRESHOLD_DEFAULT)
    pd.DataFrame([metrics_t050]).to_csv(trainer.metric_dir / "test_T050.csv", index=False)
    
    # T = Youden
    metrics_tyouden = compute_all_metrics(test_labels, test_probs, youden_threshold)
    pd.DataFrame([metrics_tyouden]).to_csv(trainer.metric_dir / "test_TYOUDEN.csv", index=False)
    
    logger.info(f"\nTest metrics at T=0.50:")
    logger.info(f"  Accuracy: {metrics_t050['accuracy']:.4f}")
    logger.info(f"  AUROC: {metrics_t050['auroc']:.4f}")
    logger.info(f"  AUPRC: {metrics_t050['auprc']:.4f}")
    logger.info(f"  F1: {metrics_t050['f1']:.4f}")
    
    logger.info(f"\nTest metrics at T={youden_threshold:.4f}:")
    logger.info(f"  Accuracy: {metrics_tyouden['accuracy']:.4f}")
    logger.info(f"  AUROC: {metrics_tyouden['auroc']:.4f}")
    logger.info(f"  AUPRC: {metrics_tyouden['auprc']:.4f}")
    logger.info(f"  F1: {metrics_tyouden['f1']:.4f}")
    
    # ========================================================================
    # 4. Per-domain metrics
    # ========================================================================
    
    logger.info("\nComputing per-domain metrics...")
    
    per_domain_results = []
    
    for threshold_name, threshold_val in [('T050', Config.THRESHOLD_DEFAULT), 
                                          ('TYOUDEN', youden_threshold)]:
        domain_metrics = compute_per_domain_metrics(
            test_labels, test_probs, test_domains, threshold_val
        )
        domain_metrics['threshold_type'] = threshold_name
        per_domain_results.append(domain_metrics)
    
    per_domain_df = pd.concat(per_domain_results, ignore_index=True)
    per_domain_df.to_csv(trainer.metric_dir / "per_domain_breakdown.csv", index=False)
    
    logger.info(f"Saved per-domain metrics to {trainer.metric_dir}")
    
    # ========================================================================
    # 5. SAVE RAW DATA for future graphing
    # ========================================================================
    
    logger.info("\nSaving raw data for graph regeneration...")
    
    # Save ROC curve data
    save_raw_curve_data(
        test_labels, test_probs,
        trainer.raw_data_dir / "roc_curve_data.csv",
        curve_type="roc"
    )
    
    # Save PR curve data
    save_raw_curve_data(
        test_labels, test_probs,
        trainer.raw_data_dir / "pr_curve_data.csv",
        curve_type="pr"
    )
    
    # Save confusion matrix data for both thresholds
    for threshold_name, threshold_val in [('T050', Config.THRESHOLD_DEFAULT),
                                          ('TYOUDEN', youden_threshold)]:
        preds = (test_probs >= threshold_val).astype(int)
        cm = confusion_matrix(test_labels, preds)
        cm_df = pd.DataFrame(cm, 
                            index=['True_Normal', 'True_Cancer'],
                            columns=['Pred_Normal', 'Pred_Cancer'])
        cm_df.to_csv(trainer.raw_data_dir / f"confusion_matrix_{threshold_name}.csv")
    
    # Save per-domain predictions for domain-specific analysis
    per_domain_preds = []
    for domain_idx in np.unique(test_domains):
        mask = (test_domains == domain_idx)
        domain_df = pd.DataFrame({
            'domain': Config.DOMAIN_NAMES.get(domain_idx, f"domain_{domain_idx}"),
            'y_true': test_labels[mask],
            'y_prob': test_probs[mask]
        })
        per_domain_preds.append(domain_df)
    
    pd.concat(per_domain_preds, ignore_index=True).to_csv(
        trainer.raw_data_dir / "per_domain_predictions.csv", index=False
    )
    
    # Save confidence distribution data
    confidence_bins = np.linspace(0, 1, 21)  # 20 bins
    confidence_hist, _ = np.histogram(test_probs, bins=confidence_bins)
    confidence_df = pd.DataFrame({
        'bin_start': confidence_bins[:-1],
        'bin_end': confidence_bins[1:],
        'count': confidence_hist
    })
    confidence_df.to_csv(trainer.raw_data_dir / "confidence_distribution.csv", index=False)
    
    # Save calibration data (for future calibration plots)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(test_probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_data = []
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            mean_prob = test_probs[mask].mean()
            mean_true = test_labels[mask].mean()
            count = mask.sum()
            calibration_data.append({
                'bin': i,
                'mean_predicted_prob': mean_prob,
                'mean_true_freq': mean_true,
                'count': count
            })
    
    pd.DataFrame(calibration_data).to_csv(
        trainer.raw_data_dir / "calibration_data.csv", index=False
    )
    
    logger.info(f"Saved raw data to {trainer.raw_data_dir}")
    
    # ========================================================================
    # 6. Generate required figures (300 DPI)
    # ========================================================================
    
    logger.info("\nGenerating figures...")
    generate_figures(trainer, metrics_t050, metrics_tyouden, 
                    test_labels, test_probs, youden_threshold)
    
    # ========================================================================
    # 7. Export deployment package
    # ========================================================================
    
    logger.info("\nExporting deployment package...")
    export_deployment_package(trainer, youden_threshold)
    
    logger.info("\nEvaluation complete!")


def generate_figures(
    trainer: Trainer,
    metrics_t050: Dict,
    metrics_tyouden: Dict,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    youden_threshold: float
):
    """Generate all required figures at 300 DPI."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = Config.DPI
    plt.rcParams['savefig.dpi'] = Config.DPI
    
    # ========================================================================
    # 1. Learning curves
    # ========================================================================
    
    history_df = pd.DataFrame(trainer.history)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history_df['epoch'], history_df['train_acc'], 
                label='Train', marker='o', markersize=3)
    axes[0].plot(history_df['epoch'], history_df['val_acc'], 
                label='Val', marker='s', markersize=3)
    axes[0].axvline(Config.PHASE1_EPOCHS, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(Config.PHASE2_EPOCHS, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history_df['epoch'], history_df['train_loss'], 
                label='Train', marker='o', markersize=3)
    axes[1].plot(history_df['epoch'], history_df['val_loss'], 
                label='Val', marker='s', markersize=3)
    axes[1].axvline(Config.PHASE1_EPOCHS, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(Config.PHASE2_EPOCHS, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss vs Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(trainer.figure_dir / "LEARNING_CURVES.png", dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 2. ROC curve
    # ========================================================================
    
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auroc = metrics_t050['auroc']
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(trainer.figure_dir / "ROC_TEST.png", dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 3. PR curve
    # ========================================================================
    
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    auprc = metrics_t050['auprc']
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f'AUPRC = {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(trainer.figure_dir / "PR_TEST.png", dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 4. Confusion matrices
    # ========================================================================
    
    for threshold_name, threshold_val, metrics in [
        ('T050', Config.THRESHOLD_DEFAULT, metrics_t050),
        ('TYOUDEN', youden_threshold, metrics_tyouden)
    ]:
        preds = (test_probs >= threshold_val).astype(int)
        cm = confusion_matrix(test_labels, preds)
        
        plt.figure(figsize=(6, 5))
        
        # Percentages
        cm_pct = cm.astype('float') / cm.sum() * 100
        
        # Annotations
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=['Normal', 'Cancer'],
                   yticklabels=['Normal', 'Cancer'],
                   cbar_kws={'label': 'Count'})
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix (T={threshold_val:.3f})\n'
                 f'Acc={metrics["accuracy"]:.3f}, F1={metrics["f1"]:.3f}')
        
        plt.savefig(trainer.figure_dir / f"CM_{threshold_name}.png", 
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()


def export_deployment_package(trainer: Trainer, youden_threshold: float):
    """Export model and config for deployment/GUI."""
    
    # Save state dict
    torch.save(
        trainer.model.state_dict(),
        trainer.deploy_dir / "model_state_dict.pth"
    )
    
    # TorchScript (optional but useful)
    try:
        dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(trainer.device)
        traced = torch.jit.trace(trainer.model, (dummy_input, 0.0))
        torch.jit.save(traced, trainer.deploy_dir / "model_torchscript.pt")
    except Exception as e:
        trainer.logger.warning(f"Could not export TorchScript: {e}")
    
    # Inference config
    inference_config = {
        "model": {
            "name": Config.MODEL_NAME,
            "architecture": "ViT-B/16 + DANN",
            "num_classes": Config.NUM_CLASSES,
            "num_domains": Config.NUM_DOMAINS
        },
        "preprocessing": {
            "image_size": Config.IMG_SIZE,
            "resize": 256,
            "center_crop": Config.IMG_SIZE,
            "normalization": {
                "mean": Config.IMAGENET_MEAN,
                "std": Config.IMAGENET_STD
            }
        },
        "class_mapping": {
            "0": "normal",
            "1": "cancer"
        },
        "thresholds": {
            "default": Config.THRESHOLD_DEFAULT,
            "youden": float(youden_threshold)
        },
        "usage": (
            "1. Load image and apply same preprocessing as eval transforms\n"
            "2. Pass through model to get logits\n"
            "3. Apply softmax to get probabilities\n"
            "4. Use probability for class 1 (cancer) and compare to threshold\n"
            "5. Output: predicted_class, probability"
        )
    }
    
    with open(trainer.deploy_dir / "inference_config.json", 'w') as f:
        json.dump(inference_config, f, indent=2)
    
    trainer.logger.info(f"Deployment package saved to {trainer.deploy_dir}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_single_seed(seed: int, data_root: Path, output_root: Path, resume: bool = False):
    """Run complete pipeline for a single seed."""
    
    # Setup
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_dir = output_root / f"VIT_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(run_dir / "logs", name=f"seed_{seed}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING RUN: SEED {seed}")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {run_dir}")
    
    # Save environment info
    save_environment_info(run_dir / "logs" / "environment.txt")
    
    # ========================================================================
    # 1. Data preparation
    # ========================================================================
    
    splits_dir = output_root.parent / "splits"
    
    if not (splits_dir / "train_12k.csv").exists():
        logger.info("\nCreating dataset splits...")
        
        # Scan dataset
        logger.info("Scanning dataset...")
        master_df = scan_dataset(data_root)
        logger.info(f"Found {len(master_df)} images")
        logger.info(f"Domains: {master_df['domain'].value_counts().to_dict()}")
        logger.info(f"Labels: {master_df['label'].value_counts().to_dict()}")
        
        # Create splits
        train_df, val_df, test_df = create_stratified_splits(master_df, splits_dir)
        logger.info("Splits created successfully")
    else:
        logger.info("\nLoading existing splits...")
        train_df = pd.read_csv(splits_dir / "train_12k.csv")
        val_df = pd.read_csv(splits_dir / "val_1200.csv")
        test_df = pd.read_csv(splits_dir / "test_600.csv")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # ========================================================================
    # 2. Create datasets and loaders
    # ========================================================================
    
    logger.info("\nCreating datasets...")
    
    train_transform, eval_transform = get_transforms()
    
    train_dataset = CancerDataset(train_df, transform=train_transform)
    val_dataset = CancerDataset(val_df, transform=eval_transform)
    test_dataset = CancerDataset(test_df, transform=eval_transform)
    
    train_sampler = create_balanced_sampler(train_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # ========================================================================
    # 3. Create model
    # ========================================================================
    
    logger.info("\nCreating model...")
    model = ViTDANN(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        num_domains=Config.NUM_DOMAINS,
        pretrained=True
    )
    logger.info(f"Model: {Config.MODEL_NAME}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # 4. Train
    # ========================================================================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=run_dir,
        seed=seed,
        logger=logger
    )
    
    trainer.train(resume=resume)
    
    # ========================================================================
    # 5. Evaluate
    # ========================================================================
    
    evaluate_and_save(trainer, val_df, test_df, logger)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SEED {seed} COMPLETE")
    logger.info(f"{'='*60}\n")


def aggregate_results(output_root: Path, logger: logging.Logger):
    """Aggregate results across all seeds."""
    
    logger.info("\n" + "="*60)
    logger.info("AGGREGATING RESULTS ACROSS SEEDS")
    logger.info("="*60)
    
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for seed in Config.SEEDS:
        run_dir = output_root / f"VIT_seed_{seed}"
        
        # Load test metrics
        for threshold_name in ['T050', 'TYOUDEN']:
            metrics_file = run_dir / "metrics" / f"test_{threshold_name}.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                df['seed'] = seed
                df['threshold_type'] = threshold_name
                all_metrics.append(df)
    
    # Combine and compute mean ± std
    combined = pd.concat(all_metrics, ignore_index=True)
    
    # Group by threshold type
    summary = combined.groupby('threshold_type').agg({
        'accuracy': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'auroc': ['mean', 'std'],
        'auprc': ['mean', 'std']
    })
    
    summary.to_csv(summary_dir / "VIT_test_thresholds_mean_std.csv")
    
    logger.info(f"\nAggregated results saved to {summary_dir}")
    logger.info("\nSummary:")
    logger.info(summary.to_string())
    
    # Also aggregate per-domain metrics
    all_domain_metrics = []
    
    for seed in Config.SEEDS:
        run_dir = output_root / f"VIT_seed_{seed}"
        domain_file = run_dir / "metrics" / "per_domain_breakdown.csv"
        if domain_file.exists():
            df = pd.read_csv(domain_file)
            df['seed'] = seed
            all_domain_metrics.append(df)
    
    if all_domain_metrics:
        combined_domain = pd.concat(all_domain_metrics, ignore_index=True)
        
        domain_summary = combined_domain.groupby(['domain', 'threshold_type']).agg({
            'accuracy': ['mean', 'std'],
            'auroc': ['mean', 'std'],
            'auprc': ['mean', 'std'],
            'f1': ['mean', 'std']
        })
        
        domain_summary.to_csv(summary_dir / "VIT_per_domain_mean_std.csv")
        logger.info(f"\nPer-domain summary saved")


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="ViT-B/16 + DANN Training Pipeline")
    parser.add_argument('--data_root', type=str, default="/content/drive/MyDrive/CS4/DATASET",
                       help="Path to dataset root")
    parser.add_argument('--output_root', type=str, default="runs",
                       help="Path to output directory")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                       help="Seeds to run")
    parser.add_argument('--resume', action='store_true',
                       help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Setup main logger
    main_logger = setup_logger(output_root, name="main")
    
    main_logger.info("="*60)
    main_logger.info("ViT-B/16 + DANN TRAINING PIPELINE")
    main_logger.info("="*60)
    main_logger.info(f"Data root: {data_root}")
    main_logger.info(f"Output root: {output_root}")
    main_logger.info(f"Seeds: {args.seeds}")
    main_logger.info(f"Resume: {args.resume}")
    
    # Update config if custom seeds provided
    if args.seeds != Config.SEEDS:
        Config.SEEDS = args.seeds
    
    # Run each seed
    for seed in Config.SEEDS:
        try:
            run_single_seed(seed, data_root, output_root, resume=args.resume)
        except Exception as e:
            main_logger.error(f"Error in seed {seed}: {e}", exc_info=True)
            continue
    
    # Aggregate results
    try:
        aggregate_results(output_root, main_logger)
    except Exception as e:
        main_logger.error(f"Error aggregating results: {e}", exc_info=True)
    
    main_logger.info("\n" + "="*60)
    main_logger.info("PIPELINE COMPLETE")
    main_logger.info("="*60)


if __name__ == "__main__":
    main()
