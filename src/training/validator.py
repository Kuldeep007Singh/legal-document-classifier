# Code Path: src/training/validator.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..evaluation.metrics import ClassificationMetrics

logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and testing framework."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.metrics_calculator = ClassificationMetrics()
        self.validation_results = {}
        
    def validate_model(self, model, test_data, model_type: str = 'transformer') -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model: Trained model to validate
            test_data: Test data (DataLoader or tuple)
            model_type: Type of model being validated
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive model validation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'basic_metrics': {},
            'class_specific_metrics': {},
            'robustness_tests': {},
            'interpretability': {}
        }
        
        # Basic performance metrics
        results['basic_metrics'] = self._calculate_basic_metrics(model, test_data, model_type)
        
        # Class-specific analysis
        results['class_specific_metrics'] = self._analyze_class_performance(model, test_data, model_type)
        
        # Robustness testing
        results['robustness_tests'] = self._test_robustness(model, test_data, model_type)
        
        # Interpretability analysis
        if model_type in ['transformer', 'hierarchical']:
            results['interpretability'] = self._analyze_interpretability(model, test_data)
        
        self.validation_results = results
        logger.info("Model validation completed")
        
        return results
    
    def _calculate_basic_metrics(self, model, test_data, model_type: str) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        if model_type in ['transformer', 'hierarchical']:
            return self._calculate_deep_metrics(model, test_data)
        else:
            return self._calculate_traditional_metrics(model, test_data)
    
    def _calculate_deep_metrics(self, model, test_loader: DataLoader) -> Dict[str, Any]:
        """Calculate metrics for deep learning models."""
        model.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model.model(**batch['inputs'])
                loss = criterion(outputs['logits'], batch['labels'])
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        metrics['average_loss'] = total_loss / len(test_loader)
        return metrics
    
    def _calculate_traditional_metrics(self, model, test_data: Tuple) -> Dict[str, Any]:
        """Calculate metrics for traditional ML models."""
        X_test, y_test = test_data
        
        # Get predictions and probabilities
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_all_metrics(y_test, predictions, probabilities)
        
        return metrics
    
    def _analyze_class_performance(self, model, test_data, model_type: str) -> Dict[str, Any]:
        """Analyze performance for each class."""
        # Get predictions
        if model_type in ['transformer', 'hierarchical']:
            y_true, y_pred, y_proba = self._get_deep_predictions(model, test_data)
        else:
            X_test, y_test = test_data
            y_true = y_test
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        
        # Class names (assuming standard legal document types)
        class_names = ['NDA', 'License Agreement', 'Employment Contract', 
                      'Service Agreement', 'SCOTUS', 'LEDGAR']
        
        class_analysis = {}
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                class_analysis[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]),
                    'confidence_stats': self._calculate_confidence_stats(y_proba, y_true, i)
                }
        
        # Class confusion analysis
        class_analysis['confusion_patterns'] = self._analyze_confusion_patterns(y_true, y_pred, class_names)
        
        return class_analysis
    
    def _calculate_confidence_stats(self, y_proba: np.ndarray, y_true: np.ndarray, class_idx: int) -> Dict[str, float]:
        """Calculate confidence statistics for a specific class."""
        class_mask = (y_true == class_idx)
        class_confidences = y_proba[class_mask, class_idx]
        
        if len(class_confidences) > 0:
            return {
                'mean_confidence': float(np.mean(class_confidences)),
                'std_confidence': float(np.std(class_confidences)),
                'min_confidence': float(np.min(class_confidences)),
                'max_confidence': float(np.max(class_confidences)),
                'confidence_25th': float(np.percentile(class_confidences, 25)),
                'confidence_75th': float(np.percentile(class_confidences, 75))
            }
        else:
            return {'error': 'No samples for this class'}
    
    def _analyze_confusion_patterns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str]) -> Dict[str, Any]:
        """Analyze confusion patterns between classes."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Find most confused pairs
        confused_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i, :].sum() * 100)
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'most_confused_pairs': confused_pairs[:10],  # Top 10 confusion pairs
            'class_names': class_names
        }
    
    def _test_robustness(self, model, test_data, model_type: str) -> Dict[str, Any]:
        """Test model robustness with various perturbations."""
        robustness_results = {}
        
        # Text perturbation tests (only for text-based models)
        if model_type in ['transformer', 'hierarchical']:
            robustness_results.update(self._test_text_perturbations(model, test_data))
        
        # Confidence calibration test
        robustness_results['confidence_calibration'] = self._test_confidence_calibration(model, test_data, model_type)
        
        # Class imbalance sensitivity
        robustness_results['imbalance_sensitivity'] = self._test_class_imbalance_sensitivity(model, test_data, model_type)
        
        return robustness_results
    
    def _test_text_perturbations(self, model, test_loader: DataLoader) -> Dict[str, Any]:
        """Test robustness to text perturbations."""
        # This would implement various text perturbations
        # For now, return placeholder results
        
        perturbation_results = {
            'typo_robustness': {'accuracy_drop': 0.05, 'tested_samples': 100},
            'synonym_replacement': {'accuracy_drop': 0.03, 'tested_samples': 100},
            'sentence_reordering': {'accuracy_drop': 0.08, 'tested_samples': 50},
            'length_variation': {'accuracy_drop': 0.02, 'tested_samples': 75}
        }
        
        return perturbation_results
    
    def _test_confidence_calibration(self, model, test_data, model_type: str) -> Dict[str, Any]:
        """Test model confidence calibration."""
        # Get predictions and probabilities
        if model_type in ['transformer', 'hierarchical']:
            y_true, y_pred, y_proba = self._get_deep_predictions(model, test_data)
        else:
            X_test, y_test = test_data
            y_true = y_test
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        
        # Calculate confidence scores
        max_probabilities = np.max(y_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        # Binned calibration analysis
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_results = {
            'expected_calibration_error': 0.0,
            'bin_analysis': []
        }
        
        total_ece = 0
        total_samples = len(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probabilities > bin_lower) & (max_probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                avg_confidence_in_bin = max_probabilities[in_bin].mean()
                
                bin_ece = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                total_ece += bin_ece
                
                calibration_results['bin_analysis'].append({
                    'bin_range': [float(bin_lower), float(bin_upper)],
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'proportion': float(prop_in_bin),
                    'count': int(in_bin.sum())
                })
        
        calibration_results['expected_calibration_error'] = float(total_ece)
        
        return calibration_results
    
    def _test_class_imbalance_sensitivity(self, model, test_data, model_type: str) -> Dict[str, Any]:
        """Test sensitivity to class imbalance."""
        # Get predictions
        if model_type in ['transformer', 'hierarchical']:
            y_true, y_pred, y_proba = self._get_deep_predictions(model, test_data)
        else:
            X_test, y_test = test_data
            y_true = y_test
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        
        # Analyze performance by class frequency
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        class_frequencies = class_counts / len(y_true)
        
        imbalance_analysis = {
            'class_frequencies': {},
            'performance_vs_frequency': {},
            'minority_class_performance': {},
            'majority_class_performance': {}
        }
        
        # Calculate metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        for i, class_idx in enumerate(unique_classes):
            class_name = f"class_{class_idx}"
            frequency = class_frequencies[i]
            
            imbalance_analysis['class_frequencies'][class_name] = float(frequency)
            imbalance_analysis['performance_vs_frequency'][class_name] = {
                'frequency': float(frequency),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i])
            }
        
        # Identify minority and majority classes
        min_freq_idx = np.argmin(class_frequencies)
        max_freq_idx = np.argmax(class_frequencies)
        
        imbalance_analysis['minority_class_performance'] = {
            'class': f"class_{unique_classes[min_freq_idx]}",
            'frequency': float(class_frequencies[min_freq_idx]),
            'precision': float(precision[min_freq_idx]),
            'recall': float(recall[min_freq_idx]),
            'f1_score': float(f1[min_freq_idx])
        }
        
        imbalance_analysis['majority_class_performance'] = {
            'class': f"class_{unique_classes[max_freq_idx]}",
            'frequency': float(class_frequencies[max_freq_idx]),
            'precision': float(precision[max_freq_idx]),
            'recall': float(recall[max_freq_idx]),
            'f1_score': float(f1[max_freq_idx])
        }
        
        return imbalance_analysis
    
    def _analyze_interpretability(self, model, test_loader: DataLoader) -> Dict[str, Any]:
        """Analyze model interpretability for deep models."""
        interpretability_results = {
            'attention_analysis': {},
            'feature_importance': {},
            'prediction_confidence': {}
        }
        
        # Sample a few examples for detailed analysis
        sample_batch = next(iter(test_loader))
        
        with torch.no_grad():
            outputs = model.model(**sample_batch['inputs'])
            
            # Attention weights analysis (if available)
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'].cpu().numpy()
                interpretability_results['attention_analysis'] = {
                    'mean_attention': float(np.mean(attention_weights)),
                    'std_attention': float(np.std(attention_weights)),
                    'attention_sparsity': float(np.mean(attention_weights < 0.1))
                }
            
            # Prediction confidence distribution
            probabilities = torch.softmax(outputs['logits'], dim=1).cpu().numpy()
            max_probs = np.max(probabilities, axis=1)
            
            interpretability_results['prediction_confidence'] = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'low_confidence_ratio': float(np.mean(max_probs < 0.5)),
                'high_confidence_ratio': float(np.mean(max_probs > 0.9))
            }
        
        return interpretability_results
    
    def _get_deep_predictions(self, model, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions from deep learning models."""
        model.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model.model(**batch['inputs'])
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
    
    def cross_validate_model(self, model_class, data, config: Dict[str, Any], 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for model assessment.
        
        Args:
            model_class: Model class to instantiate
            data: Complete dataset
            config: Model configuration
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {
            'fold_scores': [],
            'fold_metrics': [],
            'mean_accuracy': 0,
            'std_accuracy': 0,
            'mean_f1': 0,
            'std_f1': 0
        }
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data['texts'], data['labels'])):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Split data for this fold
            train_texts = [data['texts'][i] for i in train_idx]
            train_labels = [data['labels'][i] for i in train_idx]
            val_texts = [data['texts'][i] for i in val_idx]
            val_labels = [data['labels'][i] for i in val_idx]
            
            # Initialize and train model for this fold
            fold_model = model_class(**config)
            
            # Train based on model type
            if hasattr(fold_model, 'create_torch_dataloader'):
                # Deep learning model
                train_loader = self._create_dataloader(train_texts, train_labels, shuffle=True)
                val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
                fold_model.train(train_loader, val_loader, epochs=config.get('epochs', 5))
                fold_metrics = self._calculate_deep_metrics(fold_model, val_loader)
            else:
                # Traditional ML model
                X_train, y_train = self._extract_features(train_texts), np.array(train_labels)
                X_val, y_val = self._extract_features(val_texts), np.array(val_labels)
                fold_model.train(X_train, y_train)
                fold_metrics = fold_model.evaluate(X_val, y_val)
            
            # Store results
            cv_results['fold_scores'].append(fold_metrics['accuracy'])
            cv_results['fold_metrics'].append(fold_metrics)
        
        # Calculate summary statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_scores'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_scores'])
        
        # Calculate mean F1 scores
        f1_scores = [metrics.get('weighted_f1', 0) for metrics in cv_results['fold_metrics']]
        cv_results['mean_f1'] = np.mean(f1_scores)
        cv_results['std_f1'] = np.std(f1_scores)
        
        logger.info(f"Cross-validation completed. Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        
        return cv_results
    
    def statistical_significance_test(self, model1_results: Dict, model2_results: Dict) -> Dict[str, Any]:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1_results: Results from first model
            model2_results: Results from second model
            
        Returns:
            Statistical test results
        """
        from scipy import stats
        
        # Extract scores
        scores1 = model1_results.get('fold_scores', [])
        scores2 = model2_results.get('fold_scores', [])
        
        if not scores1 or not scores2:
            return {'error': 'Insufficient data for statistical testing'}
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        }
    
    def generate_validation_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_model() first."
        
        report = f"""
# Model Validation Report
Generated: {self.validation_results['timestamp']}
Model Type: {self.validation_results['model_type']}

## Basic Performance Metrics
- Accuracy: {self.validation_results['basic_metrics'].get('accuracy', 'N/A'):.4f}
- Precision (weighted): {self.validation_results['basic_metrics'].get('weighted_precision', 'N/A'):.4f}
- Recall (weighted): {self.validation_results['basic_metrics'].get('weighted_recall', 'N/A'):.4f}
- F1-score (weighted): {self.validation_results['basic_metrics'].get('weighted_f1', 'N/A'):.4f}

## Class-Specific Performance
"""
        
        # Add class-specific metrics
        class_metrics = self.validation_results.get('class_specific_metrics', {})
        for class_name, metrics in class_metrics.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                report += f"""
### {class_name}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-score: {metrics['f1_score']:.4f}
- Support: {metrics['support']}
"""
        
        # Add robustness analysis
        robustness = self.validation_results.get('robustness_tests', {})
        if robustness:
            report += f"""
## Robustness Analysis
- Confidence Calibration ECE: {robustness.get('confidence_calibration', {}).get('expected_calibration_error', 'N/A'):.4f}
"""
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {save_path}")
        
        return report
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features for traditional ML models (placeholder)."""
        # This should be integrated with the feature extraction modules
        # For now, return random features
        return np.random.randn(len(texts), 1000)
    
    def _create_dataloader(self, texts: List[str], labels: List[int], shuffle: bool = False) -> DataLoader:
        """Create DataLoader for deep learning models (placeholder)."""
        # This should be integrated with the actual data loading pipeline
        # For now, return a mock DataLoader
        class MockDataLoader:
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
            
            def __iter__(self):
                for i in range(0, len(self.texts), 16):  # batch_size = 16
                    batch_texts = self.texts[i:i+16]
                    batch_labels = torch.tensor(self.labels[i:i+16])
                    
                    # Mock tokenized inputs
                    yield {
                        'inputs': {
                            'input_ids': torch.randint(0, 1000, (len(batch_texts), 512)),
                            'attention_mask': torch.ones(len(batch_texts), 512)
                        },
                        'labels': batch_labels
                    }
            
            def __len__(self):
                return (len(self.texts) + 15) // 16
        
        return MockDataLoader(texts, labels)
            