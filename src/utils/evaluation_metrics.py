import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, log_loss, brier_score_loss
)
import os
from itertools import cycle

def calculate_all_metrics(y_true, y_pred, y_prob, class_names=None, output_dir=None, model_name="model"):
    """
    Calculate and optionally save all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: Names of the classes
        output_dir: Directory to save results
        model_name: Name of the model
        
    Returns:
        Dictionary containing all metrics
    """
    if class_names is None:
        num_classes = len(np.unique(y_true))
        class_names = [f"Class {i}" for i in range(num_classes)]
    else:
        num_classes = len(class_names)
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Create one-hot encoded version of y_true for ROC AUC
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate ROC AUC
    try:
        roc_auc_macro = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        roc_auc_macro = 0
        roc_auc_weighted = 0
    
    # Calculate average precision score
    try:
        avg_precision_macro = average_precision_score(y_true_onehot, y_prob, average='macro')
        avg_precision_weighted = average_precision_score(y_true_onehot, y_prob, average='weighted')
    except ValueError:
        avg_precision_macro = 0
        avg_precision_weighted = 0
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate log loss and Brier score
    try:
        log_loss_val = log_loss(y_true_onehot, y_prob)
        brier_score = brier_score_loss(y_true_onehot.ravel(), y_prob.ravel())
    except ValueError:
        log_loss_val = 0
        brier_score = 0
    
    # Compile all metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'avg_precision_macro': avg_precision_macro,
        'avg_precision_weighted': avg_precision_weighted,
        'kappa': kappa,
        'mcc': mcc,
        'log_loss': log_loss_val,
        'brier_score': brier_score,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() if not isinstance(v, np.ndarray)})
        metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)
        
        # Save per-class metrics
        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1 Score': f1_per_class
        })
        per_class_df.to_csv(os.path.join(output_dir, f"{model_name}_per_class_metrics.csv"), index=False)
        
        # Save confusion matrix
        np.savetxt(os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"), cm, delimiter=',')
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, f"{model_name}_classification_report.csv"))
        
        # Create visualizations
        create_visualizations(y_true, y_pred, y_prob, class_names, output_dir, model_name)
    
    return metrics

def create_visualizations(y_true, y_pred, y_prob, class_names, output_dir, model_name):
    """
    Create and save visualizations for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: Names of the classes
        output_dir: Directory to save visualizations
        model_name: Name of the model
    """
    num_classes = len(class_names)
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Create one-hot encoded version of y_true for ROC AUC
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    
    for i, color, name in zip(range(num_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of {name} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    
    # Compute Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        avg_precision[i] = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
    
    # Compute micro-average Precision-Recall curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_prob.ravel())
    avg_precision["micro"] = average_precision_score(y_true_onehot.ravel(), y_prob.ravel())
    
    # Plot Precision-Recall curves
    plt.plot(recall["micro"], precision["micro"],
            label=f'micro-average PR curve (AP = {avg_precision["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    for i, color, name in zip(range(num_classes), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'PR curve of {name} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_precision_recall_curve.png"))
    plt.close()
    
    # 4. Per-class metrics bar chart
    plt.figure(figsize=(12, 6))
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1 Score': f1_per_class
    }, index=class_names)
    
    # Plot
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Per-class Metrics - {model_name}')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim([0, 1.05])
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_per_class_metrics.png"))
    plt.close()
    
    # 5. Prediction distribution
    plt.figure(figsize=(10, 6))
    
    # Count occurrences of each class in true and predicted labels
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    
    # Create DataFrame for plotting
    counts_df = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    }, index=class_names)
    
    # Plot
    counts_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Class Distribution - {model_name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_class_distribution.png"))
    plt.close()