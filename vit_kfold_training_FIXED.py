#!/usr/bin/env python3
"""
FIXED K-Fold Cross-Validation Training for ViT
- Proper train/val/test split (no data leakage)
- Independent test set (never used in K-fold)
- Stratified K-fold on train+val only
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# FIXED Dataset Class - Loads ALL data once
# ============================================================================
class CancerDataset(Dataset):
    """Load cancer images from multiple sources WITHOUT pre-splitting"""
    def __init__(self, root, transform=None, img_size=224, split_type='train'):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.split_type = split_type
        
        # Load from BOTH datasets
        for dataset_name in ['LC25000', 'NCT_CRC_HE_100K']:
            dataset_path = os.path.join(root, dataset_name)
            if not os.path.exists(dataset_path):
                print(f"⚠ Warning: {dataset_name} not found at {dataset_path}")
                continue
            
            print(f"Loading from {dataset_name}...")
            
            for label_name, label in [('cancerous', 1), ('non_cancerous', 0)]:
                class_dir = os.path.join(dataset_path, label_name)
                if os.path.exists(class_dir):
                    imgs = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                    for img in imgs:
                        self.samples.append(os.path.join(class_dir, img))
                        self.labels.append(label)
                    print(f"  {label_name}: {len(imgs)} images")
        
        print(f"✓ Loaded {len(self.samples)} total images ({sum(self.labels)} cancer, {len(self.labels)-sum(self.labels)} normal)")
        
        # Define transforms based on split type
        if self.transform is None:
            if split_type == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:  # val or test
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# Model Definition
# ============================================================================
def create_vit_model(num_classes=2, pretrained=True):
    import timm
    
    class ViTClassifier(nn.Module):
        def __init__(self, model_name="vit_base_patch16_224.augreg_in21k", num_classes=2):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            embed_dim = self.backbone.num_features
            self.classifier = nn.Linear(embed_dim, num_classes)
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    return ViTClassifier(num_classes=num_classes)

# ============================================================================
# Training Function
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

# ============================================================================
# Validation Function
# ============================================================================
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

# ============================================================================
# FIXED K-Fold Training Function
# ============================================================================
def train_kfold(args):
    """
    CORRECT K-FOLD LOGIC:
    1. Load ALL data once (no pre-splitting)
    2. Split into train_val (80%) and test (20%) - test is NEVER touched
    3. K-fold ONLY on train_val set
    4. After all folds, evaluate on test set
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    print(f"K-Folds: {args.n_folds}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STEP 1: Load ALL data (no pre-splitting)
    # ========================================================================
    print("Loading complete dataset...")
    full_dataset = CancerDataset(
        root=args.data_root,
        img_size=args.img_size,
        split_type='train'  # We'll handle splits manually
    )
    
    # Get all labels for stratification
    all_labels = np.array(full_dataset.labels)
    all_indices = np.arange(len(full_dataset))
    
    # ========================================================================
    # STEP 2: Create INDEPENDENT test set (20%)
    # ========================================================================
    from sklearn.model_selection import train_test_split
    
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=args.seed
    )
    
    train_val_labels = all_labels[train_val_idx]
    test_labels = all_labels[test_idx]
    
    print(f"\n{'='*80}")
    print(f"Data Split Summary:")
    print(f"  Train+Val: {len(train_val_idx)} images")
    print(f"    - Cancer: {np.sum(train_val_labels == 1)}")
    print(f"    - Normal: {np.sum(train_val_labels == 0)}")
    print(f"  Test (held out): {len(test_idx)} images")
    print(f"    - Cancer: {np.sum(test_labels == 1)}")
    print(f"    - Normal: {np.sum(test_labels == 0)}")
    print(f"{'='*80}\n")
    
    # Create test dataset/loader (will NOT be used until final evaluation)
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ========================================================================
    # STEP 3: K-Fold Cross-Validation on train_val ONLY
    # ========================================================================
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(train_val_idx, train_val_labels)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{args.n_folds}")
        print(f"{'='*80}")
        
        # Map back to original indices
        train_indices = train_val_idx[train_idx_fold]
        val_indices = train_val_idx[val_idx_fold]
        
        train_labels_fold = all_labels[train_indices]
        val_labels_fold = all_labels[val_indices]
        
        print(f"Train: {len(train_indices)} images (Cancer: {np.sum(train_labels_fold==1)}, Normal: {np.sum(train_labels_fold==0)})")
        print(f"Val:   {len(val_indices)} images (Cancer: {np.sum(val_labels_fold==1)}, Normal: {np.sum(val_labels_fold==0)})")
        
        # Create datasets with proper transforms
        train_dataset_fold = Subset(full_dataset, train_indices)
        val_dataset_fold = Subset(full_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset_fold,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset_fold,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create model
        model = create_vit_model(num_classes=2, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        best_val_acc = 0
        best_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                fold_dir = os.path.join(args.output_dir, f'fold_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(fold_dir, 'best_model.pth'))
        
        print(f"\n✓ Fold {fold+1} Complete - Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'history': history
        })
        
        # Save fold results
        fold_dir = os.path.join(args.output_dir, f'fold_{fold+1}')
        with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
            json.dump({
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'train_size': len(train_indices),
                'val_size': len(val_indices)
            }, f, indent=2)
    
    # ========================================================================
    # STEP 4: Aggregate K-Fold Results
    # ========================================================================
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    val_accs = [r['best_val_acc'] for r in fold_results]
    print(f"Validation Accuracies across folds:")
    for i, acc in enumerate(val_accs, 1):
        print(f"  Fold {i}: {acc:.2f}%")
    print(f"\nMean Val Acc: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
    
    # Save aggregated results
    with open(os.path.join(args.output_dir, 'kfold_summary.json'), 'w') as f:
        json.dump({
            'n_folds': args.n_folds,
            'fold_results': fold_results,
            'mean_val_acc': float(np.mean(val_accs)),
            'std_val_acc': float(np.std(val_accs)),
            'train_val_size': len(train_val_idx),
            'test_size': len(test_idx)
        }, f, indent=2)
    
    # ========================================================================
    # STEP 5: Final Evaluation on Held-Out Test Set
    # ========================================================================
    print(f"\n{'='*80}")
    print("EVALUATING ON HELD-OUT TEST SET")
    print(f"{'='*80}")
    
    # Load best model from best fold
    best_fold_idx = np.argmax(val_accs)
    best_fold_path = os.path.join(args.output_dir, f'fold_{best_fold_idx+1}', 'best_model.pth')
    
    model = create_vit_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(best_fold_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"\nTest Set Results (using best fold model):")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_accuracy': float(test_acc),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'best_fold_used': best_fold_idx + 1,
            'test_size': len(test_idx)
        }, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {args.output_dir}")

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIXED K-Fold Cross-Validation for ViT')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing datasets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    train_kfold(args)
