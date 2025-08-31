# Code Path: src/training/trainer.py
import pandas as pd 
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from datetime import datetime
import yaml
from sklearn.utils import resample
import joblib

from ..models.transformer_model import LegalBERTClassifier
from ..models.traditional_ml_models import TraditionalMLClassifier, MultiModelEnsemble
from ..models.hierarchal_models import HierarchicalDocumentClassifier
from ..data.data_loader import LegalDatasetLoader
from ..evaluation.metrics import ClassificationMetrics
from .validator import ModelValidator
from .callbacks import TrainingCallbacks

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates training for different types of models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model_type = config.get('model_type', 'transformer')
        self.experiment_name = config.get('experiment_name', 'legal_classification')
        self.run_name = config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Initialize components
        self.data_loader = LegalDatasetLoader(config.get('data_config', {}))
        self.validator = ModelValidator(config.get('validation_config', {}))
        self.metrics = ClassificationMetrics()
        self.callbacks = TrainingCallbacks(config.get('callbacks_config', {}))
        
        # MLflow setup
        self.setup_mlflow()
        
        # Model and training state
        self.model = None
        self.training_history = {}
        
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        
        # Log configuration
        mlflow.log_params(self.config)
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data loaders using ALL datasets."""
        logger.info("Preparing data from all datasets...")
    
    # Load all datasets
        all_datasets = self.data_loader.load_all_datasets()
        print(f"Available datasets: {list(all_datasets.keys())}")
    
    # Combine all datasets from all splits
        all_texts = []
        all_labels = []
        dataset_sources = []
        
        for dataset_name, dataset in all_datasets.items():
            print(f"Loading dataset: {dataset_name}")
            
            # Each dataset has 'train', 'validation', 'test' splits
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    split_data = dataset[split_name]
                    print(f"  Processing {split_name} split with keys: {list(split_data.keys())}")
                    
                    # Find text and label columns in this split
                    text_key = None
                    label_key = None
                    
                    # Common text column names
                    for key in ['texts', 'text', 'content', 'document']:
                        if key in split_data:
                            text_key = key
                            break
                    
                    # Common label column names
                    for key in ['labels', 'label', 'target', 'category', 'class']:
                        if key in split_data:
                            label_key = key
                            break
                    
                    if text_key and label_key:
                        split_texts = split_data[text_key]
                        split_labels = split_data[label_key]
                        
                        # Add to combined data
                        all_texts.extend(split_texts)
                        all_labels.extend(split_labels)
                        dataset_sources.extend([f"{dataset_name}_{split_name}"] * len(split_texts))
                        
                        print(f"    Added {len(split_texts)} samples from {dataset_name} {split_name}")
                    else:
                        print(f"    Warning: Could not find text/label columns in {dataset_name} {split_name}")
                        print(f"    Available keys: {list(split_data.keys())}")
        
        # Check if we have any data
        total_samples = len(all_texts)
        if total_samples == 0:
            raise ValueError("No data found! Check that your datasets have the correct text/label column names.")
        
        unique_labels = len(set(all_labels))
        
        print(f"\nCombined dataset statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Unique labels: {unique_labels}")
        print(f"  Sample label distribution: {dict(pd.Series(all_labels).value_counts().head())}")
                
       
                # After combining all datasets, add this mapping step:
        from collections import Counter

        print(f"Before filtering: {len(all_labels)} samples, {len(set(all_labels))} unique labels")
        grouped_labels = self.group_legal_labels(all_labels, dataset_sources)
    
    # Replace fine-grained labels with grouped labels
        all_labels = grouped_labels
        
        print(f"  After grouping: {len(set(all_labels))} broad categories")
        print(f"  Categories: {sorted(set(all_labels))}")
        

        # Count label frequencies
        label_counts = Counter(all_labels)

        # Set higher threshold
        min_samples_per_class = 100  # Increased from 2

        # Map rare classes to 'Other' instead of removing them
        mapped_labels = []
        for label in all_labels:
            if label_counts[label] >= min_samples_per_class:
                mapped_labels.append(label)
            else:
                mapped_labels.append('Other')  # Group all rare classes together

        # Count after mapping
        final_label_counts = Counter(mapped_labels)
        print(f"After mapping rare classes to 'Other': {len(mapped_labels)} samples, {len(set(mapped_labels))} unique labels")
        print(f"Label distribution: {dict(final_label_counts.most_common(10))}")

        # Create combined dataset with mapped labels
        combined_data = {
            'texts': all_texts,
            'labels': mapped_labels,  # Use mapped labels
            'sources': dataset_sources
        }

        # Now split with mapped labels (should work without stratify errors)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            combined_data['texts'], 
            combined_data['labels'], 
            test_size=0.3, 
            random_state=42, 
            stratify=combined_data['labels']  # This should work now
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, 
            temp_labels, 
            test_size=0.5, 
            random_state=42, 
            stratify=temp_labels
        )

        train_data = {'texts': train_texts, 'labels': train_labels}
        val_data = {'texts': val_texts, 'labels': val_labels}
        test_data = {'texts': test_texts, 'labels': test_labels}


        
        
        print(f"\nData split completed:")
        print(f"  Train: {len(train_data['texts'])} samples")
        print(f"  Validation: {len(val_data['texts'])} samples")
        print(f"  Test: {len(test_data['texts'])} samples")
        
        # Create data loaders based on model type
        if self.model_type in ['transformer', 'hierarchical']:
            train_loader = self.data_loader.create_torch_dataloader(
                train_data, batch_size=self.config.get('batch_size', 16), shuffle=True
            )
            val_loader = self.data_loader.create_torch_dataloader(
                val_data, batch_size=self.config.get('batch_size', 16), shuffle=False
            )
            test_loader = self.data_loader.create_torch_dataloader(
                test_data, batch_size=self.config.get('batch_size', 16), shuffle=False
            )
        else:
            # For traditional ML models
            train_loader = self._prepare_traditional_data(train_data)
            val_loader = self._prepare_traditional_data(val_data)
            test_loader = self._prepare_traditional_data(test_data)
        
        return train_loader, val_loader, test_loader
    
    def group_legal_labels(self, all_labels, dataset_sources):
        """Group labels into around 10 broad legal categories."""
        
        mapped_labels = []
        for i, label in enumerate(all_labels):
            source = dataset_sources[i].lower()
            label_str = str(label).lower()
            
            # 10 broad legal categories based on your datasets
            if 'scotus' in source:
                mapped_labels.append('Supreme Court Cases')
            elif 'ledgar' in source:
                mapped_labels.append('Contracts & Agreements')
            elif 'eurlex' in source:
                mapped_labels.append('EU Regulations')
            elif 'ecthr' in source:
                mapped_labels.append('Human Rights Cases')
            elif 'unfair' in source or 'tos' in source:
                mapped_labels.append('Terms of Service')
            elif 'cuad' in source:
                mapped_labels.append('Contract Analysis')
            elif 'maud' in source:
                mapped_labels.append('Merger Agreements')
            elif 'criminal' in label_str or 'crime' in source:
                mapped_labels.append('Criminal Law')
            elif 'civil' in label_str or source:
                mapped_labels.append('Civil Law')
            elif 'corporate' in label_str or source:
                mapped_labels.append('Corporate Law')
            else:
                mapped_labels.append('Other Legal Documents')
        
        print(f"Grouped using dataset sources → {len(set(mapped_labels))} broad categories")
        print(f"Categories found: {sorted(set(mapped_labels))}")
        return mapped_labels




    
    def _prepare_traditional_data(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for traditional ML models with TF-IDF features."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Initialize TF-IDF if not exists
        if not hasattr(self, 'tfidf_vectorizer'):
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words='english',
                ngram_range=(1, 2)
            )
            # Fit on training data
            X = self.tfidf_vectorizer.fit_transform(data['texts'])
            
            joblib.dump(self.tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
            print("✅ TF-IDF vectorizer saved!")
        else:
            # Transform using existing vectorizer
            X = self.tfidf_vectorizer.transform(data['texts'])
        
        y = np.array(data['labels'])
        return X.toarray(), y  # Convert sparse matrix to dense

    
    def initialize_model(self):
        """Initialize the model based on configuration."""
        model_config = self.config.get('model_config', {})
        
        if self.model_type == 'transformer':
            self.model = LegalBERTClassifier(
                model_name=model_config.get('model_name', 'nlpaueb/legal-bert-base-uncased'),
                num_classes=model_config.get('num_classes', 6),
                dropout=model_config.get('dropout', 0.1)
            )
        elif self.model_type == 'hierarchical':
            self.model = HierarchicalDocumentClassifier(
                model_name=model_config.get('model_name', 'nlpaueb/legal-bert-base-uncased'),
                num_classes=model_config.get('num_classes', 6),
                max_sections=model_config.get('max_sections', 20),
                max_section_length=model_config.get('max_section_length', 512)
            )
        elif self.model_type == 'traditional':
            algorithm = model_config.get('algorithm', 'random_forest')
            self.model = TraditionalMLClassifier(model_type=algorithm, **model_config)
        elif self.model_type == 'ensemble':
            model_types = model_config.get('model_types', ['random_forest', 'svm', 'logistic'])
            self.model = MultiModelEnsemble(model_types=model_types)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for {self.model_type} model...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Initialize model
        self.initialize_model()
        
        # Execute training based on model type
        if self.model_type in ['transformer', 'hierarchical']:
            results = self._train_deep_model(train_loader, val_loader, test_loader)
        else:
            results = self._train_traditional_model(train_loader, val_loader, test_loader)
        
        # Final evaluation
        final_metrics = self.evaluate_model(test_loader)
        results['final_metrics'] = final_metrics
        
        # Save model
        self.save_model()
        
        # Log results to MLflow
        self.log_results(results)
        
        logger.info("Training completed successfully!")
        return results
    
    def _train_deep_model(self, train_loader, val_loader, test_loader) -> Dict[str, Any]:
        """Train transformer or hierarchical models."""
        training_config = self.config.get('training_config', {})
        
        # Training parameters
        epochs = training_config.get('epochs', 10)
        learning_rate = training_config.get('learning_rate', 2e-5)
        
        # Train the model
        if hasattr(self.model, 'train'):
            history = self.model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        else:
            # Custom training loop for hierarchical model
            history = self._custom_training_loop(train_loader, val_loader, epochs, learning_rate)
        
        return {'training_history': history}
    
    def _train_traditional_model(self, train_data, val_data, test_data) -> Dict[str, Any]:
        """Train traditional ML models."""
        X_train, y_train = train_data
        X_val, y_val = val_data if val_data else (None, None)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        if isinstance(self.model, MultiModelEnsemble):
            results = self.model.train_ensemble(X_train, y_train, validation_data)
        else:
            results = self.model.train(X_train, y_train, validation_data)
        
        return results
    
    def _custom_training_loop(self, train_loader, val_loader, epochs, learning_rate):
        """Custom training loop for models that need it."""
        # Implementation would depend on specific model requirements
        # This is a placeholder for complex training scenarios
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training and validation logic here
            # This would be implemented based on specific model needs
            pass
        
        return history
    
    def evaluate_model(self, test_loader) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        logger.info("Evaluating model on test data...")
        
        if self.model_type in ['transformer', 'hierarchical']:
            return self._evaluate_deep_model(test_loader)
        else:
            return self._evaluate_traditional_model(test_loader)
    
    def _evaluate_deep_model(self, test_loader) -> Dict[str, Any]:
        """Evaluate deep learning models."""
        self.model.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model.model(**batch['inputs'])
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(
            np.array(all_labels), 
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        
        return metrics
    
    def _evaluate_traditional_model(self, test_data) -> Dict[str, Any]:
        """Evaluate traditional ML models."""
        X_test, y_test = test_data
        
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(X_test, y_test)
        else:
            predictions = self.model.predict(X_test)
            probabilities = self.model.predict_proba(X_test)
            
            metrics = self.metrics.calculate_all_metrics(y_test, predictions, probabilities)
            return metrics
    
    def save_model(self):
        """Save the trained model."""
        model_dir = self.config.get('model_save_dir', 'models/best_models')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f"{self.model_type}_{timestamp}")
        
        if self.model_type in ['transformer', 'hierarchical']:
            model_path += '.pth'
            self.model.save_model(model_path)
        else:
            model_path += '.joblib'
            self.model.save_model(model_path)
        
        # Save configuration
        config_path = model_path.replace('.pth', '_config.yaml').replace('.joblib', '_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Model and configuration saved to {model_path}")
        return model_path
    
    def log_results(self, results: Dict[str, Any]):
        """Log results to MLflow."""
        # Log metrics
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)
        
        # Log training history
        if 'training_history' in results:
            history = results['training_history']
            for metric_name, values in history.items():
                if isinstance(values, list):
                    for epoch, value in enumerate(values):
                        mlflow.log_metric(f"{metric_name}_epoch", value, step=epoch)
        
        # Log model
        if self.model_type in ['transformer', 'hierarchical']:
            mlflow.pytorch.log_model(self.model.model, "model")
        else:
            mlflow.sklearn.log_model(self.model.pipeline if hasattr(self.model, 'pipeline') else self.model, "model")
    
    def cross_validate(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Load all data
        data = self.data_loader.load_all_data()
        
        cv_results = {
            'fold_scores': [],
            'mean_score': 0,
            'std_score': 0,
            'fold_reports': []
        }
        
        # Perform cross-validation
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data['texts'], data['labels'])):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data for this fold
            train_texts = [data['texts'][i] for i in train_idx]
            train_labels = [data['labels'][i] for i in train_idx]
            val_texts = [data['texts'][i] for i in val_idx]
            val_labels = [data['labels'][i] for i in val_idx]
            
            # Initialize fresh model for this fold
            self.initialize_model()
            
            # Train on fold
            if self.model_type in ['transformer', 'hierarchical']:
                # Create data loaders for this fold
                train_fold_loader = self.data_loader.create_torch_dataloader(
                    {'texts': train_texts, 'labels': train_labels},
                    batch_size=self.config.get('batch_size', 16),
                    shuffle=True
                )
                val_fold_loader = self.data_loader.create_torch_dataloader(
                    {'texts': val_texts, 'labels': val_labels},
                    batch_size=self.config.get('batch_size', 16),
                    shuffle=False
                )
                
                # Train model
                self.model.train(train_fold_loader, val_fold_loader, epochs=self.config.get('epochs', 5))
                
                # Evaluate
                fold_metrics = self._evaluate_deep_model(val_fold_loader)
            else:
                # Traditional ML
                X_train, y_train = self._prepare_traditional_data({'texts': train_texts, 'labels': train_labels})
                X_val, y_val = self._prepare_traditional_data({'texts': val_texts, 'labels': val_labels})
                
                self.model.train(X_train, y_train)
                fold_metrics = self.model.evaluate(X_val, y_val)
            
            fold_score = fold_metrics.get('accuracy', 0)
            cv_results['fold_scores'].append(fold_score)
            cv_results['fold_reports'].append(fold_metrics)
            
            logger.info(f"Fold {fold + 1} accuracy: {fold_score:.4f}")
        
        # Calculate overall CV metrics
        cv_results['mean_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        
        logger.info(f"Cross-validation completed. Mean accuracy: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def hyperparameter_search(self, param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters and results
        """
        logger.info("Starting hyperparameter search...")
        
        if self.model_type == 'traditional':
            # Use scikit-learn GridSearchCV for traditional models
            train_loader, val_loader, _ = self.prepare_data()
            X_train, y_train = train_loader
            
            self.initialize_model()
            results = self.model.hyperparameter_tuning(X_train, y_train, param_grid)
            
        else:
            # Custom hyperparameter search for deep models
            results = self._deep_model_hyperparameter_search(param_grid)
        
        logger.info("Hyperparameter search completed")
        return results
    
    def _deep_model_hyperparameter_search(self, param_grid: Dict) -> Dict[str, Any]:
        """Hyperparameter search for deep models."""
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = 0
        best_params = {}
        all_results = []
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            logger.info(f"Testing parameters: {params}")
            
            # Update config with current parameters
            current_config = self.config.copy()
            current_config['model_config'].update(params)
            
            # Initialize model with current parameters
            self.config = current_config
            self.initialize_model()
            
            # Perform cross-validation
            cv_results = self.cross_validate(cv_folds=3)  # Reduced folds for speed
            score = cv_results['mean_score']
            
            all_results.append({
                'params': params,
                'score': score,
                'std': cv_results['std_score']
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def cleanup(self):
        """Cleanup resources and end MLflow run."""
        mlflow.end_run()
        logger.info("Training session cleanup completed")

class ExperimentRunner:
    """Runs multiple experiments with different configurations."""
    
    def __init__(self, experiments_config: str):
        """
        Initialize experiment runner.
        
        Args:
            experiments_config: Path to experiments configuration file
        """
        with open(experiments_config, 'r') as f:
            self.experiments = yaml.safe_load(f)
        
        self.results = {}
    
    def run_all_experiments(self):
        """Run all configured experiments."""
        for exp_name, exp_config in self.experiments.items():
            logger.info(f"Running experiment: {exp_name}")
            
            try:
                trainer = ModelTrainer(exp_config)
                results = trainer.train()
                self.results[exp_name] = results
                trainer.cleanup()
                
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                self.results[exp_name] = {'error': str(e)}
        
        return self.results
    
    def compare_experiments(self) -> Dict[str, Any]:
        """Compare results across experiments."""
        comparison = {
            'experiment_names': list(self.results.keys()),
            'accuracies': [],
            'best_experiment': None,
            'best_accuracy': 0
        }
        
        for exp_name, results in self.results.items():
            if 'error' not in results:
                accuracy = results.get('final_metrics', {}).get('accuracy', 0)
                comparison['accuracies'].append(accuracy)
                
                if accuracy > comparison['best_accuracy']:
                    comparison['best_accuracy'] = accuracy
                    comparison['best_experiment'] = exp_name
        
        return comparison
    
if __name__ == "__main__":
    # Example configuration
    config = {
        'model_type': 'traditional',  # or 'transformer'
        'experiment_name': 'legal_classification_test',
        'batch_size': 16,
        'epochs': 5,
        'data_config': {},
        'model_config': {
            'num_classes': 6,
            'algorithm': 'random_forest'
        },
        'training_config': {
            'learning_rate': 2e-5,
            'epochs': 5
        },
        'model_save_dir': 'models/best_models'
    }
    
    print("Starting trainer...")
    trainer = ModelTrainer(config)
    results = trainer.train()
    print(f"Training completed! Results: {results}")
    trainer.cleanup()