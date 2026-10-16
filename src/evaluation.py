"""
Comprehensive Evaluation Suite for Multilabel and Binary Toxic Comment Models.
Calculates per-label metrics, aggregate multilabel metrics (Hamming Loss, Subset Accuracy),
ROC-AUC, Precision-Recall curves, and confusion matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    hamming_loss, accuracy_score,
    confusion_matrix, classification_report
)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str] = LABELS
) -> Dict[str, Any]:
    """
    Evaluates multilabel predictions with both per-label and aggregate metrics.
    """
    per_label_metrics = {}
    roc_auc_list = []
    pr_auc_list = []

    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prb = y_prob[:, i]

        p = float(precision_score(yt, yp, zero_division=0))
        r = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))

        # Check if positive samples exist in test split for AUC
        if len(np.unique(yt)) > 1:
            auc = float(roc_auc_score(yt, prb))
            pr_auc = float(average_precision_score(yt, prb))
        else:
            auc = 0.5
            pr_auc = 0.0

        roc_auc_list.append(auc)
        pr_auc_list.append(pr_auc)

        cm = confusion_matrix(yt, yp)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        per_label_metrics[name] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'pr_auc': round(pr_auc, 4),
            'support': int(yt.sum()),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }

    # Multilabel Aggregate Metrics
    h_loss = float(hamming_loss(y_true, y_pred))
    subset_acc = float(accuracy_score(y_true, y_pred))  # Exact match ratio
    
    micro_p = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    micro_r = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))

    macro_p = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    mean_roc_auc = float(np.mean(roc_auc_list))

    results = {
        'aggregate': {
            'mean_roc_auc': round(mean_roc_auc, 4),
            'hamming_loss': round(h_loss, 4),
            'subset_accuracy': round(subset_acc, 4),
            'micro_precision': round(micro_p, 4),
            'micro_recall': round(micro_r, 4),
            'micro_f1': round(micro_f1, 4),
            'macro_precision': round(macro_p, 4),
            'macro_recall': round(macro_r, 4),
            'macro_f1': round(macro_f1, 4),
            'weighted_f1': round(weighted_f1, 4)
        },
        'per_label': per_label_metrics
    }

    return results


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates binary toxic/non-toxic predictions.
    """
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': round(acc, 4),
        'precision': round(p, 4),
        'recall': round(r, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'specificity': round(specificity, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    steps: int = 100
) -> Dict[str, Any]:
    """
    Finds the optimal decision threshold that maximizes F1 or balances precision & recall.
    """
    thresholds = np.linspace(0.05, 0.95, steps)
    best_th = 0.5
    best_score = -1.0
    history = []

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        score = f1 if metric == 'f1' else (2 * p * r / (p + r) if (p + r) > 0 else 0)
        history.append({'threshold': round(float(th), 3), 'precision': round(float(p), 3), 'recall': round(float(r), 3), 'f1': round(float(f1), 3)})

        if score > best_score:
            best_score = score
            best_th = float(th)

    return {
        'optimal_threshold': round(best_th, 3),
        'best_score': round(float(best_score), 4),
        'curve': history
    }
