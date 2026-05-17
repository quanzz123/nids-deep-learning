import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

REPORTS_DIR = 'reports'


def evaluate(model, X_test, y_test):
    """Compute Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_probs[:, 1]),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return results, y_pred


def plot_loss_curve(history):
    """Plot and save training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=1.5)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=1.5)
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=1.5)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=1.5)
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{REPORTS_DIR}/loss_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_confusion_matrix(cm):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = f'{REPORTS_DIR}/confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_feature_importance(selected_features, importances_dict, save_path=None):
    """Plot and save selected feature importances."""
    names = selected_features
    scores = [importances_dict[n] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(range(len(names)), scores, color='mediumseagreen')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance')
    plt.title('Selected Feature Importances (ETC)')
    plt.gca().invert_yaxis()

    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{score:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    path = save_path or f'{REPORTS_DIR}/feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def save_report_txt(results, selected_features, importances_dict, model=None):
    """Save evaluation metrics, confusion matrix, selected features, and model summary to report.txt."""
    cm = results['confusion_matrix']
    lines = []
    lines.append('=' * 50)
    lines.append('  NIDS EVALUATION REPORT')
    lines.append('=' * 50)
    lines.append('')
    lines.append('--- Metrics ---')
    for key, val in results.items():
        if key == 'confusion_matrix':
            continue
        key_str = key.replace('_', ' ').title()
        lines.append(f'  {key_str:<20s}  {val:.4f}')
    lines.append('')
    lines.append('--- Confusion Matrix ---')
    lines.append(f'  TN={cm[0,0]}  FP={cm[0,1]}')
    lines.append(f'  FN={cm[1,0]}  TP={cm[1,1]}')
    lines.append('')
    lines.append('--- Selected Features ---')
    for feat in selected_features:
        lines.append(f'  {feat}: {importances_dict[feat]:.6f}')
    lines.append('')
    lines.append('=' * 50)

    # Model summary (optional)
    if model is not None:
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + '\n'))
        lines.append('')
        lines.append('--- Model Summary ---')
        lines.append(buf.getvalue())
        lines.append('=' * 50)

    path = f'{REPORTS_DIR}/report.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {path}')


def print_metrics_table(results):
    """Print evaluation metrics in a formatted table."""
    print()
    print('=' * 45)
    print('  EVALUATION METRICS')
    print('=' * 45)
    for key, val in results.items():
        if key == 'confusion_matrix':
            continue
        key_str = key.replace('_', ' ').title()
        print(f'  {key_str:<20s}  {val:.4f}')
    print('=' * 45)
    print()
    print('Confusion Matrix:')
    print(f'  {results["confusion_matrix"]}')
    print()
