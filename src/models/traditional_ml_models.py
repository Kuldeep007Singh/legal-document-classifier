# Code Path: src/models/traditional_models.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class TraditionalMLClassifier(BaseModel):
    """Traditional machine learning models for legal document classification."""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize traditional ML classifier.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'logistic', 'gradient_boost', 'naive_bayes')
            **kwargs: Model-specific parameters
        """
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.pipeline = None
        self.feature_selector = None
        self.model_params = kwargs
        
        # Default hyperparameters
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,        # Back to original
                'max_depth': 10,          # Back to unlimited depth
                'min_samples_split': 20,     # Back to original
                'min_samples_leaf': 10, 
                'max_features': 'sqrt',     # Back to original
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',           # Back to RBF kernel
                'gamma': 'scale',
                'random_state': 42
            },
            'logistic': {
                'C': 1.0,
                'max_iter': 1000,          # Back to 1000 iterations
                'random_state': 42
            },
            'gradient_boost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'naive_bayes': {
                'alpha': 1.0
            }
        }


        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        
        # Define allowed parameters for each sklearn model
        allowed_params_map = {
            'random_forest': set(RandomForestClassifier().get_params().keys()),
            'svm': set(SVC().get_params().keys()), 
            'logistic': set(LogisticRegression().get_params().keys()),
            'gradient_boost': set(GradientBoostingClassifier().get_params().keys()),
            'naive_bayes': {'alpha'}  # MultinomialNB params
        }
        
        # Get allowed parameters for this model type
        allowed_params = allowed_params_map.get(self.model_type, set())
        
        # Filter out invalid parameters from self.model_params
        clean_model_params = {k: v for k, v in self.model_params.items() 
                            if k in allowed_params}
        
        # Filter defaults too
        clean_defaults = {k: v for k, v in self.default_params.get(self.model_type, {}).items() 
                        if k in allowed_params}
        
        # Merge defaults with cleaned params
        final_params = {**clean_defaults, **clean_model_params}
        
        # Initialize models with cleaned parameters
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**final_params)
        elif self.model_type == 'svm':
            # Ensure probability=True for SVM
            final_params['probability'] = True
            self.model = SVC(**final_params)
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(**final_params)
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(**final_params)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**final_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    
  # Import f_classif

    def create_pipeline(self, feature_selection: bool = True, k_features: int = 1000):
        """Create a scikit-learn pipeline with feature selection."""
        steps = []
        
        if feature_selection:
            self.feature_selector = SelectKBest(f_classif, k= 'all')  # Use f_classif instead of chi2
            steps.append(('feature_selection', self.feature_selector))
        
        steps.append(('classifier', self.model))
        self.pipeline = Pipeline(steps)
        
        return self.pipeline

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
            validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the traditional ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Training metrics and history
        """
        print(f"\n🔄 Training {self.model_type} model with {len(X_train):,} samples...")
        
        # Create pipeline if not exists
        if self.pipeline is None:
            self.create_pipeline()
        
        # Train the model
        print("🔄 Fitting model...")
        self.pipeline.fit(X_train, y_train)
        print("✅ Model fitting completed!")
        
        # Evaluate on training data
        train_score = self.pipeline.score(X_train, y_train)
        
        # Store minimal metrics (no verbose classification_report)
        metrics = {
            'train_accuracy': train_score,
        }
        
        # Evaluate on validation data if provided
        val_score = None
        if validation_data:
            X_val, y_val = validation_data
            print("🔄 Evaluating on validation set...")
            val_score = self.pipeline.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            print("✅ Validation evaluation completed!")
        
        # **Use custom cross-validation with progress**
        cv_scores = self.cross_validate_with_progress(X_train, y_train, cv=5)
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
        
        # CLEAN SUMMARY OUTPUT
        print("\n" + "="*50)
        print("🎉 TRAINING RESULTS")
        print("="*50)
        print(f"📊 Model: {self.model_type.upper()}")
        print(f"📈 Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        if validation_data:
            print(f"✅ Validation Accuracy: {val_score:.4f} ({val_score*100:.2f}%)")
        print(f"🔄 Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"📋 Legal Categories: {len(set(y_train))}")
        print("="*50)
        
        return metrics

    def cross_validate_with_progress(self, X_train, y_train, cv=5):
        """Cross-validation with progress tracking."""
        from sklearn.model_selection import KFold
        import numpy as np
        from sklearn.base import clone
        
        print(f"🔄 Starting {cv}-fold cross-validation...")
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            print(f"   📊 Running CV Fold {fold}/{cv}...")
            
            # Split data for this fold
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Create a COPY of the pipeline for this fold (IMPORTANT!)
            fold_pipeline = clone(self.pipeline)
            
            # Train on this fold
            fold_pipeline.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation fold
            fold_score = fold_pipeline.score(X_fold_val, y_fold_val)
            scores.append(fold_score)
            
            print(f"   ✅ CV Fold {fold}/{cv} completed - Accuracy: {fold_score:.4f}")

        scores = np.array(scores)
        print(f"🎯 Cross-Validation Results:")
        print(f"   Individual fold scores: {[f'{s:.4f}' for s in scores]}")
        print(f"   Mean CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return scores


    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            param_grid: Optional[Dict] = None, cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
        
        # Create pipeline
        if self.pipeline is None:
            self.create_pipeline()
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.pipeline = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Get default parameter grid for hyperparameter tuning."""
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'svm': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            },
            'logistic': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            },
            'gradient_boost': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            },
            'naive_bayes': {
                'classifier__alpha': [0.1, 1.0, 10.0]
            }
        }
        
        return param_grids.get(self.model_type, {})
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the abstract fit method from BaseModel."""
        print(f"Fitting {self.model_type} model...")
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"✓ {self.model_type} model fitted successfully!")
    
    def evaluate(self, X_test, y_test):
        if self.pipeline is None:  # Use pipeline, not model
            raise ValueError("Model not trained yet. Call train() first.")
        
        accuracy = self.pipeline.score(X_test, y_test)
        return {'accuracy': accuracy}

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if self.pipeline is None:
            return None
        
        classifier = self.pipeline.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            return classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            return np.abs(classifier.coef_).flatten()
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model."""
        self.pipeline = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

        
    
    

class MultiModelEnsemble:
    """Ensemble of multiple traditional ML models."""
    
    def __init__(self, model_types: list = None):
        """
        Initialize ensemble with multiple model types.
        
        Args:
            model_types: List of model types to include in ensemble
        """
        if model_types is None:
            model_types = ['random_forest', 'svm', 'logistic', 'gradient_boost']
        
        self.model_types = model_types
        self.models = {}
        self.weights = None
        
        # Initialize models
        for model_type in model_types:
            self.models[model_type] = TraditionalMLClassifier(model_type)
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                      validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model...")
            model_results = model.train(X_train, y_train, validation_data)
            results[model_type] = model_results
        
        # Calculate ensemble weights based on validation performance
        if validation_data:
            self._calculate_weights(validation_data)
        
        return results
    
    def _calculate_weights(self, validation_data: Tuple):
        """Calculate ensemble weights based on validation performance."""
        X_val, y_val = validation_data
        accuracies = []
        
        for model in self.models.values():
            accuracy = model.pipeline.score(X_val, y_val)
            accuracies.append(accuracy)
        
        # Softmax weights
        accuracies = np.array(accuracies)
        exp_acc = np.exp(accuracies - np.max(accuracies))
        self.weights = exp_acc / np.sum(exp_acc)
        
        logger.info(f"Ensemble weights: {dict(zip(self.model_types, self.weights))}")
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            method: Ensemble method ('voting', 'weighted', 'stacking')
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        probabilities = []
        
        for model in self.models.values():
            pred = model.predict(X)
            proba = model.predict_proba(X)
            predictions.append(pred)
            probabilities.append(proba)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        if method == 'voting':
            # Majority voting
            from scipy.stats import mode
            ensemble_pred = mode(predictions, axis=0)[0].flatten()
        elif method == 'weighted' and self.weights is not None:
            # Weighted average of probabilities
            weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
            ensemble_pred = np.argmax(weighted_proba, axis=1)
        else:
            # Simple average of probabilities
            avg_proba = np.mean(probabilities, axis=0)
            ensemble_pred = np.argmax(avg_proba, axis=1)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """Get ensemble prediction probabilities."""
        probabilities = []
        
        for model in self.models.values():
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)
        
        if method == 'weighted' and self.weights is not None:
            return np.average(probabilities, axis=0, weights=self.weights)
        else:
            return np.mean(probabilities, axis=0)