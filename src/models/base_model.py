# src/models/base_model.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str = "BaseModel"):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.label_encoder = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"{self.model_name} evaluation metrics: {metrics}")
        return metrics
    
    def get_classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """Get detailed classification report"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting classification report")
        
        predictions = self.predict(X)
        return classification_report(y, predictions)
    
    def save_model(self, save_path: Path) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path) -> None:
        """Load a trained model"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data.get('feature_names')
        self.label_encoder = model_data.get('label_encoder')
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_names) if self.feature_names else None
        }

    def __str__(self) -> str:
        return f"{self.model_name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()
