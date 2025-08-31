# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ClassificationMetrics:
    """Class for computing classification metrics"""
    
    def __init__(self):
        pass
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """Compute comprehensive classification metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # ROC AUC if probabilities provided
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, 
                                                     multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        return metrics
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred)
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
