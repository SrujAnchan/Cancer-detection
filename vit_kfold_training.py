#!/usr/bin/env python3
"""
ViT K-Fold Cross-Validation Training Script
Implements K-Fold CV with K=2, 5, 10 for cancer classification
Saves comprehensive results for paper publication
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your model and dataset
try:
    # Try to import from your existing code
    from vit_model import create_vit_model
    from data_module import LC25000
except ImportError:
    # Fall back to wrapper
    from kfold_imports import create_vit_model, LC25000
    print("⚠ Using fallback imports - please check kfold_imports.py")


class KFoldTrainer:
    """Handles K-Fold cross-validation training and evaluation"""
    
    def __init__(self, args, k_value):
        self.args = args
        self.k_value = k_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory structure
        self.k_root = Path(args.output_root) / f"K{k_value}"
        self.k_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.fold_results = []
        self.fold_times = []
        self.master_log = []
        
        print(f"\n{'='*80}")
        print(f"Initializing K={k_value} Cross-Validation")
        print(f"Output directory: {self.k_root}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
    def load_data(self):
        """Load and combine train+val data for K-fold splitting"""
        print("Loading datasets...")
        
        # Load train and validation sets
        train_dst = LC25000(
            self.args.data_root,
            split='train',
            domain_label=0,
            img_size=self.args.img_size
        )
        
        val_dst = LC25000(
            self.args.data_root,
            split='val',
            domain_label=0,
            img_size=self.args.img_size
        )
        
        # Combine for K-fold splitting
        self.combined_data = torch.utils.data.ConcatDataset([train_dst, val_dst])
        
        # Get labels for stratification
        self.combined_labels = []
        for i in range(len(self.combined_data)):
            _, label, _ = self.combined_data[i]
            self.combined_labels.append(label)
        self.combined_labels = np.array(self.combined_labels)
        
        # Load fixed test set
        self.test_dst = LC25000(
            self.args.data_root,
            split='test',
            domain_label=0,
            img_size=self.args.img_size
        )
        
        print(f"✓ Combined train+val size: {len(self.combined_data)}")
        print(f"✓ Test set size: {len(self.test_dst)}")
        print(f"✓ Class distribution: {np.bincount(self.combined_labels)}")
        
        self.master_log.append({
            'event': 'data_loaded',
            'combined_size': len(self.combined_data),
            'test_size': len(self.test_dst),
            'class_distribution': np.bincount(self.combined_labels).tolist()
        })
        
    def create_fold_splits(self):
        """Create stratified K-fold splits"""
        print(f"\nCreating {self.k_value}-fold stratified splits...")
        
        skf = StratifiedKFold(
            n_splits=self.k_value,
            shuffle=True,
            random_state=self.args.seed
        )
        
        self.fold_splits = list(skf.split(
            np.arange(len(self.combined_data)),
            self.combined_labels
        ))
        
        print(f"✓ Created {len(self.fold_splits)} folds")
        
        # Log split info
        for fold_idx, (train_idx, val_idx) in enumerate(self.fold_splits):
            train_labels = self.combined_labels[train_idx]
            val_labels = self.combined_labels[val_idx]
            print(f"  Fold {fold_idx}: Train={len(train_idx)} "
                  f"(Class 0: {np.sum(train_labels==0)}, Class 1: {np.sum(train_labels==1)}), "
                  f"Val={len(val_idx)} "
                  f"(Class 0: {np.sum(val_labels==0)}, Class 1: {np.sum(val_labels==1)})")
        
    def train_single_fold(self, fold_idx, train_idx, val_idx):
        """Train model for a single fold"""
        fold_start_time = time.time()
        
        # Create fold directory
        fold_dir = self.k_root / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        checkpoints_dir = fold_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        curves_dir = fold_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        
        logs_dir = fold_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Training Fold {fold_idx}/{self.k_value-1}")
        print(f"{'='*80}")
        
        # Create data loaders
        train_subset = Subset(self.combined_data, train_idx)
        val_subset = Subset(self.combined_data, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Create model
        model = create_vit_model(num_classes=2).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epochs
        )
        
        # Training history
        history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion, epoch
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(time.time() - epoch_start)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                    },
                    checkpoints_dir / 'model_best.pth'
                )
            
            # Periodic checkpoint saves
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                    },
                    checkpoints_dir / f'model_epoch_{epoch+1}.pth'
                )
            
            # Print progress
            elapsed = time.time() - fold_start_time
            epochs_remaining = self.args.num_epochs - (epoch + 1)
            avg_epoch_time = np.mean(history['epoch_times'])
            eta = epochs_remaining * avg_epoch_time
            
            print(f"Epoch {epoch+1}/{self.args.num_epochs} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Time: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
        
        # Save final model
        torch.save(
            {
                'epoch': self.args.num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            },
            checkpoints_dir / 'model_final.pth'
        )
        
        # Save training history
        with open(logs_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Generate training curves
        self._plot_training_curves(history, curves_dir, fold_idx)
        
        # Evaluate on test set
        test_results = self._evaluate_on_test(
            model, fold_dir, fold_idx
        )
        
        fold_time = time.time() - fold_start_time
        self.fold_times.append(fold_time)
        
        # Save fold summary
        fold_summary = {
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'final_val_acc': history['val_acc'][-1],
            'training_time_seconds': fold_time,
            'test_results': test_results
        }
        
        with open(fold_dir / 'fold_summary.json', 'w') as f:
            json.dump(fold_summary, f, indent=2)
        
        self.fold_results.append(fold_summary)
        
        # Clear memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        print(f"\n✓ Fold {fold_idx} completed in {fold_time/60:.1f} minutes")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print(f"  Test Acc: {test_results['accuracy']:.4f}")
        
        return fold_summary
    
    def _train_epoch(self, model, loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(loader), 100. * correct / total
    
    def _validate_epoch(self, model, loader, criterion, epoch):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Validation Epoch {epoch+1}', leave=False)
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss / len(loader), 100. * correct / total
    
    def _evaluate_on_test(self, model, fold_dir, fold_idx):
        """Evaluate model on fixed test set"""
        print(f"\nEvaluating Fold {fold_idx} on test set...")
        
        eval_dir = fold_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Load best model
        checkpoint = torch.load(
            fold_dir / "checkpoints" / "model_best.pth",
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_loader = DataLoader(
            self.test_dst,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        # ROC AUC (for binary classification, use class 1 probabilities)
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_preds,
            target_names=['Normal', 'Cancer'],
            output_dict=True
        )
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auroc': float(auroc),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        }
        
        with open(eval_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, eval_dir, fold_idx)
        
        # Plot ROC curve
        self._plot_roc_curve(fpr, tpr, auroc, eval_dir, fold_idx)
        
        # Plot classification report
        self._plot_classification_report(class_report, eval_dir, fold_idx)
        
        print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
        print(f"✓ Test AUROC: {auroc:.4f}")
        
        return results
    
    def _plot_training_curves(self, history, curves_dir, fold_idx):
        """Generate training curve plots (2 versions as requested)"""
        
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Version 1: Normal scale (zoomed 94-100%)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot (zoomed)
        ax1.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=2)
        ax1.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', markersize=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Training & Validation Accuracy (Zoomed)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([94, 100])
        
        # Loss plot (normal scale)
        ax2.plot(epochs, history['train_loss'], 'b-o', label='Train Cancer Loss', markersize=2)
        ax2.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', markersize=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training & Validation Loss (Cancer Classification)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(curves_dir / f'training_curves_normal_fold{fold_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Version 2: Publication quality with log scale
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(epochs, history['train_acc'], 'b-o', label='Training', markersize=2, linewidth=2)
        ax1.plot(epochs, history['val_acc'], 'r-s', label='Validation', markersize=2, linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Model Accuracy', fontsize=14, fontweight='bold', loc='left')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([94, 100])
        
        # Loss plot (log scale)
        ax2.semilogy(epochs, history['train_loss'], 'b-o', label='Training', markersize=2, linewidth=2)
        ax2.semilogy(epochs, history['val_loss'], 'r-s', label='Validation', markersize=2, linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Classification Loss', fontsize=14, fontweight='bold', loc='left')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(curves_dir / f'training_curves_publication_fold{fold_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves to {curves_dir}")
    
    def _plot_confusion_matrix(self, cm, eval_dir, fold_idx):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Cancer'],
            yticklabels=['Normal', 'Cancer'],
            cbar_kws={'label': 'Count'}
        )
        
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Calculate F1 score
        precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.title(f'Confusion Matrix (T=0.50)\nAccuracy={accuracy:.4f}, F1={f1:.4f}',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(eval_dir / f'confusion_matrix_fold{fold_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, fpr, tpr, auroc, eval_dir, fold_idx):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 8))
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Cancer (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.tight_layout()
        plt.savefig(eval_dir / f'roc_curve_fold{fold_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_report(self, class_report, eval_dir, fold_idx):
        """Plot classification report as table"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        table_data.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        for class_name in ['Normal', 'Cancer']:
            metrics = class_report[class_name]
            table_data.append([
                class_name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                f"{int(metrics['support'])}"
            ])
        
        table_data.append(['', '', '', '', ''])
        table_data.append([
            'Accuracy', '', '', 
            f"{class_report['accuracy']:.4f}",
            f"{int(class_report['macro avg']['support'])}"
        ])
        table_data.append([
            'Macro Avg',
            f"{class_report['macro avg']['precision']:.4f}",
            f"{class_report['macro avg']['recall']:.4f}",
            f"{class_report['macro avg']['f1-score']:.4f}",
            f"{int(class_report['macro avg']['support'])}"
        ])
        table_data.append([
            'Weighted Avg',
            f"{class_report['weighted avg']['precision']:.4f}",
            f"{class_report['weighted avg']['recall']:.4f}",
            f"{class_report['weighted avg']['f1-score']:.4f}",
            f"{int(class_report['weighted avg']['support'])}"
        ])
        
        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color class rows
        for i in range(5):
            table[(1, i)].set_facecolor('#E8F5E9')
            table[(2, i)].set_facecolor('#FFE0B2')
        
        plt.title('Classification Report - Test Set', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(eval_dir / f'classification_report_fold{fold_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def aggregate_results(self):
        """Aggregate results across all folds"""
        print(f"\n{'='*80}")
        print(f"Aggregating results for K={self.k_value}")
        print(f"{'='*80}\n")
        
        agg_dir = self.k_root / "aggregated_results"
        agg_dir.mkdir(exist_ok=True)
        
        # Extract metrics from all folds
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auroc': []
        }
        
        for fold_result in self.fold_results:
            test_res = fold_result['test_results']
            metrics['accuracy'].append(test_res['accuracy'])
            metrics['precision'].append(test_res['precision'])
            metrics['recall'].append(test_res['recall'])
            metrics['f1_score'].append(test_res['f1_score'])
            metrics['auroc'].append(test_res['auroc'])
        
        # Calculate statistics
        stats = {}
        for metric_name, values in metrics.items():
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
        
        # Save aggregated metrics
        with open(agg_dir / 'aggregated_metrics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create comparison plots
        self._plot_fold_comparison(metrics, agg_dir)
        
        # Create summary table
        self._create_summary_table(stats, agg_dir)
        
        # Calculate total time
        total_time = sum(self.fold_times)
        avg_time_per_fold = np.mean(self.fold_times)
        
        timing_info = {
            'total_time_seconds': total_time,
            'total_time_formatted': str(timedelta(seconds=int(total_time))),
            'average_time_per_fold_seconds': avg_time_per_fold,
            'fold_times': self.fold_times
        }
        
        with open(agg_dir / 'timing_info.json', 'w') as f:
            json.dump(timing_info, f, indent=2)
        
        print(f"\n✓ K={self.k_value} Summary:")
        print(f"  Mean Accuracy: {stats['accuracy']['mean']*100:.2f}% ± {stats['accuracy']['std']*100:.2f}%")
        print(f"  Mean AUROC: {stats['auroc']['mean']:.4f} ± {stats['auroc']['std']:.4f}")
        print(f"  Total Training Time: {timedelta(seconds=int(total_time))}")
        print(f"  Average Time per Fold: {timedelta(seconds=int(avg_time_per_fold))}")
        
        return stats
    
    def _plot_fold_comparison(self, metrics, agg_dir):
        """Plot comparison of metrics across folds"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = list(metrics.keys())
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            values = metrics[metric_name]
            folds = range(len(values))
            
            ax.bar(folds, values, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.axhline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
            ax.set_xlabel('Fold', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_title(f'{metric_name.replace("_", " ").title()} per Fold', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(agg_dir / 'per_fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_table(self, stats, agg_dir):
        """Create summary table of results"""
        import pandas as pd
        
        # Create DataFrame
        data = []
        for metric_name, stat_dict in stats.items():
            data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Mean': f"{stat_dict['mean']:.4f}",
                'Std': f"{stat_dict['std']:.4f}",
                'Min': f"{stat_dict['min']:.4f}",
                'Max': f"{stat_dict['max']:.4f}"
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        df.to_csv(agg_dir / 'summary_table.csv', index=False)
        
        # Plot as table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title(f'K={self.k_value} Fold Cross-Validation Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(agg_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_kfold(self):
        """Main method to run K-fold cross-validation"""
        start_time = time.time()
        
        # Load data
        self.load_data()
        
        # Create folds
        self.create_fold_splits()
        
        # Train each fold
        for fold_idx, (train_idx, val_idx) in enumerate(self.fold_splits):
            self.train_single_fold(fold_idx, train_idx, val_idx)
            
            # Save progress after each fold
            progress = {
                'k_value': self.k_value,
                'completed_folds': fold_idx + 1,
                'total_folds': self.k_value,
                'fold_results': self.fold_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.k_root / 'progress.json', 'w') as f:
                json.dump(progress, f, indent=2)
        
        # Aggregate results
        aggregated_stats = self.aggregate_results()
        
        total_time = time.time() - start_time
        
        # Create final summary
        final_summary = {
            'k_value': self.k_value,
            'num_folds': len(self.fold_results),
            'aggregated_stats': aggregated_stats,
            'total_time_seconds': total_time,
            'total_time_formatted': str(timedelta(seconds=int(total_time))),
            'fold_results': self.fold_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.k_root / 'final_summary.json', 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"K={self.k_value} Cross-Validation COMPLETED")
        print(f"Total Time: {timedelta(seconds=int(total_time))}")
        print(f"Results saved to: {self.k_root}")
        print(f"{'='*80}\n")
        
        return final_summary


def parse_args():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation for ViT Cancer Classification')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Path to output root (will create K2, K5, K10 subdirs)')
    
    # Training arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # K-Fold arguments
    parser.add_argument('--k_values', nargs='+', type=int, default=[2, 5, 10],
                        help='K values for cross-validation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"\n{'#'*80}")
    print(f"# ViT K-Fold Cross-Validation Training")
    print(f"# K values: {args.k_values}")
    print(f"# Output directory: {args.output_root}")
    print(f"# Seed: {args.seed}")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    # Run K-fold for each K value
    for k_value in args.k_values:
        print(f"\n{'#'*80}")
        print(f"# Starting K={k_value} Cross-Validation")
        print(f"{'#'*80}\n")
        
        trainer = KFoldTrainer(args, k_value)
        result = trainer.run_kfold()
        all_results[f'K{k_value}'] = result
    
    # Save master results
    output_path = Path(args.output_root)
    with open(output_path / 'all_kfold_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"# ALL K-FOLD EXPERIMENTS COMPLETED")
    print(f"# Master results saved to: {output_path / 'all_kfold_results.json'}")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
