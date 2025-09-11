
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
     confusion_matrix, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve,
    log_loss, brier_score_loss
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



class ComprehensiveMetrics:
    """Class to calculate comprehensive metrics for binary classification"""
    
    def __init__(self, pathway_names=None):
        if pathway_names is None:
            self.pathway_names = [f'pathway_{i}' for i in range(34)]
        else:
            self.pathway_names = pathway_names
        
    def calculate_all_metrics(self, y_true, y_pred_proba, y_pred_binary=None, threshold=0.5):
        """
        Calculate comprehensive metrics for binary classification
        
        Args:
            y_true: True binary labels (n_samples, n_pathways)
            y_pred_proba: Predicted probabilities (n_samples, n_pathways)
            y_pred_binary: Predicted binary labels (optional)
            threshold: Threshold for converting probabilities to binary predictions
        
        Returns:
            Dictionary with all metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        # Calculate metrics for each pathway
        for i, pathway in enumerate(self.pathway_names):
            if i >= y_true.shape[1]:
                break
                
            y_t = y_true[:, i]
            y_p_proba = y_pred_proba[:, i]
            y_p_binary = y_pred_binary[:, i]
            
            pathway_metrics = self._calculate_pathway_metrics(y_t, y_p_proba, y_p_binary)
            metrics[pathway] = pathway_metrics
        
        # Calculate overall metrics
        metrics['overall'] = self._calculate_overall_metrics(y_true, y_pred_proba, y_pred_binary)
        
        return metrics
    
    def _calculate_pathway_metrics(self, y_true, y_pred_proba, y_pred_binary):
        """Calculate metrics for a single pathway"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['specificity'] = self._calculate_specificity(y_true, y_pred_binary)
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        # Advanced metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_binary)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred_binary)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred_binary)
        
        # Probabilistic metrics (only if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
            metrics['log_loss'] = np.nan
            metrics['brier_score'] = np.nan
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            
            # Additional derived metrics
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            metrics['false_discovery_rate'] = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        # Optimal threshold analysis
        if len(np.unique(y_true)) > 1:
            optimal_metrics = self._find_optimal_threshold(y_true, y_pred_proba)
            metrics.update(optimal_metrics)
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0
    
    def _find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold using various criteria"""
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
        
        # Youden's J statistic (maximizes TPR - FPR)
        j_scores = tpr - fpr
        optimal_idx_youden = np.argmax(j_scores)
        
        # F1 score maximization
        f1_scores = []
        for thresh in thresholds_pr:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        optimal_idx_f1 = np.argmax(f1_scores)
        
        return {
            'optimal_threshold_youden': thresholds_roc[optimal_idx_youden],
            'optimal_j_score': j_scores[optimal_idx_youden],
            'optimal_threshold_f1': thresholds_pr[optimal_idx_f1],
            'optimal_f1_score': max(f1_scores)
        }
    
    def _calculate_overall_metrics(self, y_true, y_pred_proba, y_pred_binary):
        """Calculate overall metrics across all pathways"""
        metrics = {}
        
        # Micro-averaged metrics
        metrics['micro_precision'] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        
        # Macro-averaged metrics
        metrics['macro_precision'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        
        # Weighted metrics
        metrics['weighted_precision'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        
        # Subset accuracy (exact match for multi-label)
        metrics['subset_accuracy'] = np.mean(np.all(y_true == y_pred_binary, axis=1))
        
        # Hamming loss (fraction of incorrect labels)
        metrics['hamming_loss'] = np.mean(y_true != y_pred_binary)
        
        # Label-based metrics
        metrics['total_positives'] = int(np.sum(y_true))
        metrics['total_predictions'] = int(np.sum(y_pred_binary))
        metrics['correct_predictions'] = int(np.sum(y_true * y_pred_binary))
        
        return metrics

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(history['val_f1_macro'], label='Macro F1', color='green')
    axes[0, 1].set_title('Validation Macro F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[1, 0].plot(history['val_accuracy'], label='Subset Accuracy', color='purple')
    axes[1, 0].set_title('Validation Subset Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Pathway-specific F1 scores over time
    if len(history['val_metrics']) > 0:
        pathway1_f1 = [m['pathway1']['f1_score'] for m in history['val_metrics']]
        pathway2_f1 = [m['pathway2']['f1_score'] for m in history['val_metrics']]
        
        axes[1, 1].plot(pathway1_f1, label='Pathway1 F1', color='orange')
        axes[1, 1].plot(pathway2_f1, label='Pathway2 F1', color='cyan')
        axes[1, 1].set_title('Pathway-specific F1 Scores')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrices(results, save_path=None):
    """Plot confusion matrices for each pathway"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, pathway in enumerate(['pathway1', 'pathway2']):
        y_true = results['labels'][:, i]
        y_pred = results['predictions_binary'][:, i]
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i])
        axes[i].set_title(f'{pathway.capitalize()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()

def plot_roc_curves(results, save_path=None):
    """Plot ROC curves for each pathway"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, pathway in enumerate(['pathway1', 'pathway2']):
        y_true = results['labels'][:, i]
        y_probs = results['predictions_proba'][:, i]
        
        # Only plot if both classes are present
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            auc_score = roc_auc_score(y_true, y_probs)
            
            axes[i].plot(fpr, tpr, color='blue', lw=2, 
                        label=f'ROC curve (AUC = {auc_score:.3f})')
            axes[i].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', 
                        label='Random')
            
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{pathway.capitalize()} ROC Curve')
            axes[i].legend(loc="lower right")
            axes[i].grid(True)
        else:
            axes[i].text(0.5, 0.5, 'Single class present\nROC curve not applicable', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{pathway.capitalize()} ROC Curve')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()

def analyze_dataset_performance(results, save_path=None):
    """Analyze performance by dataset"""
    predictions_df = pd.DataFrame({
        'dataset_id': results['dataset_ids'],
        'pathway1_true': results['labels'][:, 0],
        'pathway2_true': results['labels'][:, 1],
        'pathway1_pred': results['predictions_binary'][:, 0],
        'pathway2_pred': results['predictions_binary'][:, 1],
    })
    
    # Calculate metrics by dataset
    dataset_metrics = {}
    
    for dataset in predictions_df['dataset_id'].unique():
        dataset_data = predictions_df[predictions_df['dataset_id'] == dataset]
        
        # Calculate F1 for each pathway
        pathway1_f1 = f1_score(dataset_data['pathway1_true'], 
                              dataset_data['pathway1_pred'], zero_division=0)
        pathway2_f1 = f1_score(dataset_data['pathway2_true'], 
                              dataset_data['pathway2_pred'], zero_division=0)
        
        # Calculate subset accuracy
        subset_acc = np.mean((dataset_data['pathway1_true'] == dataset_data['pathway1_pred']) & 
                            (dataset_data['pathway2_true'] == dataset_data['pathway2_pred']))
        
        dataset_metrics[dataset] = {
            'n_samples': len(dataset_data),
            'pathway1_f1': pathway1_f1,
            'pathway2_f1': pathway2_f1,
            'subset_accuracy': subset_acc,
            'macro_f1': (pathway1_f1 + pathway2_f1) / 2
        }
    
    # Create DataFrame for easy viewing
    dataset_metrics_df = pd.DataFrame(dataset_metrics).T
    dataset_metrics_df = dataset_metrics_df.sort_values('macro_f1', ascending=False)
    
    print("\nPerformance by Dataset:")
    print(dataset_metrics_df.round(4))
    
    if save_path:
        dataset_metrics_df.to_csv(f"{save_path}_dataset_performance.csv")
        print(f"Dataset performance saved to {save_path}_dataset_performance.csv")
    
    return dataset_metrics_df