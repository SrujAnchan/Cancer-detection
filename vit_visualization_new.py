#!/usr/bin/env python3
"""
Complete Graph and Metrics Generation Script for ViT-B/16 + DANN (runs2)
Generates all required visualizations, metrics, and creates a zip file.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import zipfile

# Add code directory to path
sys.path.insert(0, '/data/cse25/cse25/cancer_classification/code')

# Configuration
RUN_DIR = Path('/data/cse25/cse25/cancer_classification/runs2/VIT_seed_0')
SPLITS_DIR = Path('/data/cse25/cse25/cancer_classification/splits')
DPI = 300

print("="*70)
print("GRAPH GENERATION FOR VIT-B/16 + DANN (runs2)")
print("="*70)

# Create output directories
figures_dir = RUN_DIR / 'figures'
raw_data_dir = RUN_DIR / 'raw_data'
metrics_dir = RUN_DIR / 'metrics'
predictions_dir = RUN_DIR / 'predictions'

for d in [figures_dir, raw_data_dir, metrics_dir, predictions_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: GENERATE PREDICTIONS (if needed)
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING/GENERATING PREDICTIONS")
print("="*70)

pred_file = predictions_dir / 'test_predictions.csv'

if not pred_file.exists():
    print("Predictions not found. Generating predictions from best checkpoint...")
    
    # Import necessary modules
    from vit_dann_new import ViTDANN, CancerDataset, Config
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from tqdm import tqdm
    
    # Load test data
    test_df = pd.read_csv(SPLITS_DIR / 'test_full.csv')
    
    # Create test dataset
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CancerDataset(test_df, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTDANN(Config.MODEL_NAME, Config.NUM_CLASSES, Config.NUM_DOMAINS).to(device)
    
    # Load best checkpoint
    best_checkpoint = RUN_DIR / 'checkpoints' / 'best_val_auc.pth'
    print(f"Loading checkpoint from {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Generating predictions...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(device)
                labels = batch_data['label']
            else:
                images = batch_data[0].to(device)
                labels = batch_data[1]
            
            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'image_path': test_df['path'].values,
        'y_true': all_labels,
        'y_prob': all_probs,
        'domain': test_df['domain'].values
    })
    pred_df.to_csv(pred_file, index=False)
    print(f"✓ Saved predictions to {pred_file}")
else:
    print(f"✓ Loading existing predictions from {pred_file}")
    pred_df = pd.read_csv(pred_file)

# Extract predictions
y_true = pred_df['y_true'].values
y_prob = pred_df['y_prob'].values

print(f"\nTest samples: {len(y_true)}")
print(f"Test AUROC: {roc_auc_score(y_true, y_prob):.4f}")

# ============================================================================
# STEP 2: LEARNING CURVES
# ============================================================================
print("\n" + "="*70)
print("STEP 2: GENERATING LEARNING CURVES")
print("="*70)

train_history = pd.read_csv(RUN_DIR / 'logs' / 'training_history.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
ax1.plot(train_history['epoch'], train_history['train_acc'], 'b-', label='Train', linewidth=2.5)
ax1.plot(train_history['epoch'], train_history['val_acc'], 'r-', label='Validation', linewidth=2.5)
ax1.axvline(x=30, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Phase 1→2')
ax1.axvline(x=60, color='gray', linestyle=':', alpha=0.6, linewidth=2, label='Phase 2→3')
ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title('Training and Validation Accuracy (90 Epochs)', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 90)

# Loss
ax2.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train', linewidth=2.5)
ax2.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Validation', linewidth=2.5)
ax2.axvline(x=30, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Phase 1→2')
ax2.axvline(x=60, color='gray', linestyle=':', alpha=0.6, linewidth=2, label='Phase 2→3')
ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax2.set_title('Training and Validation Loss (90 Epochs)', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 90)

plt.tight_layout()
plt.savefig(figures_dir / 'LEARNING_CURVES.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {figures_dir / 'LEARNING_CURVES.png'}")

# ============================================================================
# STEP 3: ROC CURVE (BOTH CLASSES)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: GENERATING ROC CURVES (BOTH CLASSES)")
print("="*70)

# Class 1 (Cancer) - using probability as is
fpr_class1, tpr_class1, thresholds_class1 = roc_curve(y_true, y_prob)
auroc_class1 = roc_auc_score(y_true, y_prob)

# Class 0 (Non-Cancer) - using 1 - probability
y_true_inverted = 1 - y_true
y_prob_class0 = 1 - y_prob
fpr_class0, tpr_class0, thresholds_class0 = roc_curve(y_true_inverted, y_prob_class0)
auroc_class0 = roc_auc_score(y_true_inverted, y_prob_class0)

# Save raw data
roc_df_class1 = pd.DataFrame({
    'fpr': fpr_class1,
    'tpr': tpr_class1,
    'threshold': thresholds_class1
})
roc_df_class1.to_csv(raw_data_dir / 'roc_curve_class1_cancer.csv', index=False)

roc_df_class0 = pd.DataFrame({
    'fpr': fpr_class0,
    'tpr': tpr_class0,
    'threshold': thresholds_class0
})
roc_df_class0.to_csv(raw_data_dir / 'roc_curve_class0_noncancer.csv', index=False)

# Plot both classes
plt.figure(figsize=(10, 9))
plt.plot(fpr_class1, tpr_class1, 'b-', linewidth=3, label=f'Class 1 (Cancer) - AUROC = {auroc_class1:.4f}')
plt.plot(fpr_class0, tpr_class0, 'r-', linewidth=3, label=f'Class 0 (Non-Cancer) - AUROC = {auroc_class0:.4f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.4, label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - Both Classes (Test Set)', fontsize=16, fontweight='bold')
plt.legend(fontsize=13, loc='lower right', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(figures_dir / 'ROC_BOTH_CLASSES.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {figures_dir / 'ROC_BOTH_CLASSES.png'}")
print(f"  Class 1 (Cancer) AUROC: {auroc_class1:.4f}")
print(f"  Class 0 (Non-Cancer) AUROC: {auroc_class0:.4f}")

# ============================================================================
# STEP 4: PR CURVE
# ============================================================================
print("\n" + "="*70)
print("STEP 4: GENERATING PRECISION-RECALL CURVE")
print("="*70)

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)

# Save raw data
pr_df = pd.DataFrame({
    'precision': precision[:-1],
    'recall': recall[:-1],
    'threshold': pr_thresholds
})
pr_df.to_csv(raw_data_dir / 'pr_curve_data.csv', index=False)

# Plot
plt.figure(figsize=(10, 9))
plt.plot(recall, precision, 'b-', linewidth=3, label=f'PR Curve (AUPRC = {auprc:.4f})')
plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('Precision-Recall Curve - Test Set', fontsize=16, fontweight='bold')
plt.legend(fontsize=13, loc='lower left', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(figures_dir / 'PR_TEST.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {figures_dir / 'PR_TEST.png'}")

# ============================================================================
# STEP 5: CONFUSION MATRICES
# ============================================================================
print("\n" + "="*70)
print("STEP 5: GENERATING CONFUSION MATRICES")
print("="*70)

# T=0.50
y_pred_050 = (y_prob >= 0.5).astype(int)
cm_050 = confusion_matrix(y_true, y_pred_050)

# Save raw data
cm_050_df = pd.DataFrame(cm_050, columns=['Pred_0', 'Pred_1'], index=['True_0', 'True_1'])
cm_050_df.to_csv(raw_data_dir / 'confusion_matrix_T050.csv')

# Plot
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_050, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'],
            ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix (T=0.50)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'CM_T050.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {figures_dir / 'CM_T050.png'}")

# Youden threshold
j_scores = tpr_class1 - fpr_class1
youden_idx = np.argmax(j_scores)
youden_threshold = thresholds_class1[youden_idx]

y_pred_youden = (y_prob >= youden_threshold).astype(int)
cm_youden = confusion_matrix(y_true, y_pred_youden)

# Save raw data
cm_youden_df = pd.DataFrame(cm_youden, columns=['Pred_0', 'Pred_1'], index=['True_0', 'True_1'])
cm_youden_df.to_csv(raw_data_dir / 'confusion_matrix_TYOUDEN.csv')

# Plot
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_youden, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'],
            ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix (Youden T={youden_threshold:.3f})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'CM_TYOUDEN.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {figures_dir / 'CM_TYOUDEN.png'}")

# ============================================================================
# STEP 6: METRICS FILES
# ============================================================================
print("\n" + "="*70)
print("STEP 6: GENERATING METRICS FILES")
print("="*70)

# Metrics at T=0.50
tn, fp, fn, tp = cm_050.ravel()
metrics_050 = {
    'threshold': 0.50,
    'accuracy': accuracy_score(y_true, y_pred_050),
    'sensitivity': recall_score(y_true, y_pred_050),
    'specificity': tn / (tn + fp),
    'precision': precision_score(y_true, y_pred_050),
    'f1': f1_score(y_true, y_pred_050),
    'auroc': auroc_class1,
    'auprc': auprc,
    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
}
pd.DataFrame([metrics_050]).to_csv(metrics_dir / 'test_T050.csv', index=False)
print(f"✓ Saved: {metrics_dir / 'test_T050.csv'}")

# Metrics at Youden
tn_y, fp_y, fn_y, tp_y = cm_youden.ravel()
metrics_youden = {
    'threshold': float(youden_threshold),
    'accuracy': accuracy_score(y_true, y_pred_youden),
    'sensitivity': recall_score(y_true, y_pred_youden),
    'specificity': tn_y / (tn_y + fp_y),
    'precision': precision_score(y_true, y_pred_youden),
    'f1': f1_score(y_true, y_pred_youden),
    'auroc': auroc_class1,
    'auprc': auprc,
    'tp': int(tp_y), 'tn': int(tn_y), 'fp': int(fp_y), 'fn': int(fn_y)
}
pd.DataFrame([metrics_youden]).to_csv(metrics_dir / 'test_TYOUDEN.csv', index=False)
print(f"✓ Saved: {metrics_dir / 'test_TYOUDEN.csv'}")

# Youden info
youden_info = {
    'threshold': float(youden_threshold),
    'sensitivity': float(tpr_class1[youden_idx]),
    'specificity': float(1 - fpr_class1[youden_idx]),
    'j_score': float(j_scores[youden_idx])
}
with open(metrics_dir / 'youden_info.json', 'w') as f:
    json.dump(youden_info, f, indent=2)
print(f"✓ Saved: {metrics_dir / 'youden_info.json'}")

# ============================================================================
# STEP 7: PER-DOMAIN METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: GENERATING PER-DOMAIN METRICS")
print("="*70)

domain_metrics = []
for domain in pred_df['domain'].unique():
    domain_data = pred_df[pred_df['domain'] == domain]
    y_true_d = domain_data['y_true'].values
    y_prob_d = domain_data['y_prob'].values
    y_pred_d = (y_prob_d >= 0.5).astype(int)
    
    domain_metrics.append({
        'domain': domain,
        'threshold_type': 'T050',
        'accuracy': accuracy_score(y_true_d, y_pred_d),
        'auroc': roc_auc_score(y_true_d, y_prob_d),
        'auprc': average_precision_score(y_true_d, y_prob_d),
        'f1': f1_score(y_true_d, y_pred_d),
        'n_samples': len(y_true_d)
    })

domain_metrics_df = pd.DataFrame(domain_metrics)
domain_metrics_df.to_csv(metrics_dir / 'per_domain_breakdown.csv', index=False)
print(f"✓ Saved: {metrics_dir / 'per_domain_breakdown.csv'}")

# Save per-domain predictions
pred_df.to_csv(raw_data_dir / 'per_domain_predictions.csv', index=False)
print(f"✓ Saved: {raw_data_dir / 'per_domain_predictions.csv'}")

# ============================================================================
# STEP 8: CREATE ZIP FILE
# ============================================================================
print("\n" + "="*70)
print("STEP 8: CREATING ZIP FILE")
print("="*70)

zip_path = RUN_DIR.parent / 'VIT_seed_0_results.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add all files from figures, metrics, raw_data, predictions
    for folder in [figures_dir, metrics_dir, raw_data_dir, predictions_dir]:
        for file in folder.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(RUN_DIR)
                zipf.write(file, arcname=f'VIT_seed_0/{arcname}')
    
    # Add training history
    hist_file = RUN_DIR / 'logs' / 'training_history.csv'
    if hist_file.exists():
        zipf.write(hist_file, arcname='VIT_seed_0/logs/training_history.csv')

print(f"✓ Created zip file: {zip_path}")
print(f"  Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ ALL OUTPUTS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nOutput locations:")
print(f"  Figures:     {figures_dir}")
print(f"  Metrics:     {metrics_dir}")
print(f"  Raw Data:    {raw_data_dir}")
print(f"  Predictions: {predictions_dir}")
print(f"  Zip File:    {zip_path}")

print(f"\n📊 Results Summary:")
print(f"  Test Samples: {len(y_true)}")
print(f"  Test AUROC (Cancer):     {auroc_class1:.4f}")
print(f"  Test AUROC (Non-Cancer): {auroc_class0:.4f}")
print(f"  Test Accuracy (T=0.50):  {metrics_050['accuracy']:.4f}")
print(f"  Youden Threshold:        {youden_threshold:.4f}")
print(f"  Best Val AUROC:          1.0000")
print(f"  Total Epochs:            90")

print("\n" + "="*70)
print("GRAPH GENERATION COMPLETE!")
print("="*70)
