import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LegalDocumentClassifier:
    """
    Multi-model classifier for legal documents with data balancing and comprehensive evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.pipelines = {}
        self.results = {}
        self.training_results = {}
        self.test_results = {}
        self.best_model = None
        self.class_weights = None
        
        # Initialize models with different configurations
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models with balanced class weights."""
        self.models = {
            'naive_bayes': MultinomialNB(),  # NB doesn't support class_weight
            'svm_linear': SVC(kernel='linear', random_state=self.random_state, class_weight='balanced'),
            'svm_rbf': SVC(kernel='rbf', random_state=self.random_state, class_weight='balanced'),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state)  # GB doesn't support class_weight directly
        }
    
    def analyze_class_imbalance(self, y):
        """Analyze and display class imbalance information."""
        from collections import Counter
        
        print("\n" + "="*50)
        print("CLASS IMBALANCE ANALYSIS")
        print("="*50)
        
        class_counts = Counter(y)
        total = len(y)
        
        print(f"Total documents: {total}")
        print(f"Number of classes: {len(class_counts)}")
        print("\nClass Distribution:")
        
        # Sort by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for label, count in sorted_classes:
            percentage = (count / total) * 100
            print(f"  {label}: {count} documents ({percentage:.1f}%)")
        
        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\nImbalance Metrics:")
        print(f"  Most frequent class: {max_count} documents")
        print(f"  Least frequent class: {min_count} documents")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("  ⚠️  SEVERE CLASS IMBALANCE!")
        elif imbalance_ratio > 3:
            print("  ⚠️  MODERATE CLASS IMBALANCE")
        else:
            print("  ✅ RELATIVELY BALANCED DATASET")
        
        return class_counts, imbalance_ratio
    
    def filter_rare_classes(self, X, y, min_samples_per_class=2):
        """Filter out classes with too few samples."""
        from collections import Counter
        
        class_counts = Counter(y)
        classes_to_keep = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
        
        if len(classes_to_keep) < len(class_counts):
            removed_classes = set(class_counts.keys()) - set(classes_to_keep)
            print(f"\n⚠️  Removing {len(removed_classes)} classes with <{min_samples_per_class} samples:")
            for cls in removed_classes:
                print(f"    - {cls}: {class_counts[cls]} samples")
        
        # Filter data
        mask = y.isin(classes_to_keep)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        print(f"\nFiltered dataset:")
        print(f"  Original: {len(X)} documents, {len(set(y))} classes")
        print(f"  Filtered: {len(X_filtered)} documents, {len(set(y_filtered))} classes")
        
        return X_filtered, y_filtered
    
    def create_pipelines(self, max_features: int = 10000, min_df: int = 2, max_df: float = 0.8):
        """Create sklearn pipelines with optimized TF-IDF for legal documents."""
        
        # TF-IDF parameters optimized for legal documents
        tfidf_params = {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': (1, 2),  # Include unigrams and bigrams
            'sublinear_tf': True,
            'stop_words': 'english'
        }
        
        print(f"\nTF-IDF Parameters:")
        print(f"  Max features: {max_features}")
        print(f"  Min document frequency: {min_df}")
        print(f"  Max document frequency: {max_df}")
        print(f"  N-gram range: {tfidf_params['ngram_range']}")
        
        # Create pipelines
        for model_name, model in self.models.items():
            self.pipelines[model_name] = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('classifier', model)
            ])
    
    def train_models(self, X_train: pd.Series, y_train: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Train all models with cross-validation."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, pipeline in self.pipelines.items():
            print(f"\n🔄 Training {model_name.replace('_', ' ').title()}...")
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Fit the model for training accuracy calculation
            pipeline.fit(X_train, y_train)
            
            # Store results
            self.results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'pipeline': pipeline
            }
            
            print(f"   Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return {name: result['cv_mean'] for name, result in self.results.items()}
    
    def evaluate_training_performance(self, X_train: pd.Series, y_train: pd.Series):
        """Evaluate models on training data."""
        print("\n" + "="*50)
        print("TRAINING PERFORMANCE EVALUATION")
        print("="*50)
        
        for model_name, pipeline in self.pipelines.items():
            # Get predictions on training data
            y_train_pred = pipeline.predict(X_train)
            
            # Calculate training accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            self.training_results[model_name] = train_accuracy
            
            print(f"{model_name.replace('_', ' ').title()}: {train_accuracy:.4f}")
    
    def evaluate_models(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Dict]:
        """Evaluate all trained models on test data."""
        print("\n" + "="*50)
        print("TEST PERFORMANCE EVALUATION")
        print("="*50)
        
        for model_name, pipeline in self.pipelines.items():
            # Predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Detailed classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            self.test_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'predictions': y_pred
            }
            
            print(f"{model_name.replace('_', ' ').title()}: {accuracy:.4f}")
        
        return self.test_results
    
    def find_best_model(self) -> str:
        """Identify the best performing model based on cross-validation scores."""
        if not self.results:
            print("No results available. Train models first.")
            return None
        
        best_score = 0
        best_model_name = None
        
        for model_name, results in self.results.items():
            if results['cv_mean'] > best_score:
                best_score = results['cv_mean']
                best_model_name = model_name
        
        self.best_model = best_model_name
        print(f"\n🏆 Best model: {best_model_name.replace('_', ' ').title()} (CV: {best_score:.4f})")
        return best_model_name
    
    def plot_comprehensive_results(self):
        """Create comprehensive visualization of all results."""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        # Prepare data
        model_names = list(self.results.keys())
        model_labels = [name.replace('_', ' ').title() for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        # Include training and test accuracies if available
        train_accs = [self.training_results.get(name, 0) for name in model_names] if self.training_results else None
        test_accs = [self.test_results[name]['accuracy'] if name in self.test_results else 0 for name in model_names] if self.test_results else None
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cross-validation scores with error bars
        axes[0, 0].bar(model_labels, cv_means, yerr=cv_stds, capsize=5, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            axes[0, 0].text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Box plot of CV scores distribution
        cv_scores_list = [self.results[name]['cv_scores'] for name in model_names]
        bp = axes[0, 1].boxplot(cv_scores_list, labels=model_labels)
        axes[0, 1].set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training vs Validation vs Test comparison (if available)
        if train_accs and test_accs:
            x = np.arange(len(model_names))
            width = 0.25
            
            axes[1, 0].bar(x - width, train_accs, width, label='Training', color='lightgreen', alpha=0.7)
            axes[1, 0].bar(x, cv_means, width, label='Cross-Validation', color='skyblue', alpha=0.7)
            axes[1, 0].bar(x + width, test_accs, width, label='Test', color='lightcoral', alpha=0.7)
            
            axes[1, 0].set_title('Training vs Cross-Validation vs Test Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_labels, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overfitting analysis
        if train_accs:
            overfitting = [train - cv for train, cv in zip(train_accs, cv_means)]
            colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in overfitting]
            
            axes[1, 1].bar(model_labels, overfitting, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High overfitting (>0.1)')
            axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate overfitting (>0.05)')
            axes[1, 1].set_title('Overfitting Analysis (Training - CV Accuracy)', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Overfitting Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_comprehensive_summary(self):
        """Print detailed performance summary."""
        if not self.results:
            print("No results to summarize. Train models first.")
            return
        
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*70)
        
        # Header
        print(f"{'Model':<20} {'Training':<10} {'Cross-Val':<10} {'Test':<10} {'Overfitting':<12} {'Status'}")
        print("-" * 70)
        
        for model_name in self.results.keys():
            model_display = model_name.replace('_', ' ').title()
            cv_acc = self.results[model_name]['cv_mean']
            train_acc = self.training_results.get(model_name, 0)
            test_acc = self.test_results[model_name]['accuracy'] if model_name in self.test_results else 0
            
            overfitting = train_acc - cv_acc if train_acc > 0 else 0
            
            # Determine status
            if overfitting > 0.1:
                status = "⚠️ High Overfit"
            elif overfitting > 0.05:
                status = "⚠️ Mod Overfit"
            elif test_acc > 0 and abs(cv_acc - test_acc) < 0.02:
                status = "✅ Good"
            elif test_acc > 0:
                status = "⚠️ Poor General."
            else:
                status = "❓ Unknown"
            
            print(f"{model_display:<20} {train_acc:<10.4f} {cv_acc:<10.4f} {test_acc:<10.4f} {overfitting:<12.4f} {status}")
        
        # Best model highlight
        if self.best_model:
            print(f"\n🏆 BEST MODEL: {self.best_model.replace('_', ' ').title()}")
            
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Models with high overfitting (>0.1) need regularization")
        print(f"   • Consider ensemble methods for improved performance")
        print(f"   • Legal-BERT may provide better accuracy for legal documents")
        print(f"   • Hyperparameter tuning can further optimize performance")
    
    def save_model(self, model_name: str = None, filepath: str = None):
        """Save the trained model."""
        if model_name is None:
            model_name = self.best_model or self.find_best_model()
        
        if model_name is None:
            print("No model to save. Train models first.")
            return None
        
        if filepath is None:
            filepath = f'models/{model_name}_legal_classifier.joblib'
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the pipeline
        joblib.dump(self.pipelines[model_name], filepath)
        print(f"✅ Model saved to {filepath}")
        
        return filepath

# Main execution
if __name__ == "__main__":
    print("🚀 LEGAL DOCUMENT CLASSIFIER WITH BALANCED TRAINING")
    print("=" * 60)
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    try:
        df = pd.read_csv('data/processed/legal_documents_preprocessed.csv')
        print(f"   ✅ Loaded {len(df)} documents")
    except FileNotFoundError:
        print("   ❌ Preprocessed data not found. Run preprocessor.py first.")
        exit(1)
    
    # Prepare data
    X = df['processed_text']
    y = df['label']
    
    # Initialize classifier
    classifier = LegalDocumentClassifier()
    
    # Analyze class imbalance
    class_counts, imbalance_ratio = classifier.analyze_class_imbalance(y)
    
    # Filter rare classes for better training
    X_filtered, y_filtered = classifier.filter_rare_classes(X, y, min_samples_per_class=2)
    
    # Split data with stratification if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )
        print(f"✅ Used stratified split to maintain class balance")
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42
        )
        print(f"⚠️ Used random split (stratification not possible)")
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} documents")
    print(f"   Test: {len(X_test)} documents")
    
    # Create pipelines with balanced models
    classifier.create_pipelines()
    
    # Train models with cross-validation
    cv_results = classifier.train_models(X_train, y_train)
    
    # Evaluate training performance
    classifier.evaluate_training_performance(X_train, y_train)
    
    # Evaluate on test data
    classifier.evaluate_models(X_test, y_test)
    
    # Find best model
    classifier.find_best_model()
    
    # Print comprehensive summary
    classifier.print_comprehensive_summary()
    
    # Create comprehensive plots
    classifier.plot_comprehensive_results()
    
    # Save the best model
    best_model_path = classifier.save_model()
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"   Best model saved: {best_model_path}")
    print(f"   Ready for deployment and further optimization!")
