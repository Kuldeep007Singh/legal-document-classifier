import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RegularizedLegalDocumentClassifier:
    """
    Heavily regularized classifier to combat overfitting in legal document classification.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.pipelines = {}
        self.results = {}
        self.training_results = {}
        self.test_results = {}
        
        # Initialize HEAVILY regularized models
        self._initialize_regularized_models()
    
    def _initialize_regularized_models(self):
        """Initialize models with strong regularization to prevent overfitting."""
        self.models = {
            # Naive Bayes with smoothing
            'naive_bayes_regularized': MultinomialNB(alpha=1.0),  # Strong smoothing
            
            # SVM with strong regularization
            'svm_linear_regularized': SVC(
                kernel='linear', 
                C=0.1,  # Strong regularization (low C)
                random_state=self.random_state, 
                class_weight='balanced'
            ),
            
            # Logistic Regression with L2 penalty
            'logistic_l2_regularized': LogisticRegression(
                C=0.1,  # Strong regularization
                penalty='l2',
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            # Ridge Classifier (built-in regularization)
            'ridge_classifier': RidgeClassifier(
                alpha=10.0,  # Strong regularization
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Random Forest with constraints
            'random_forest_regularized': RandomForestClassifier(
                n_estimators=50,  # Fewer trees
                max_depth=3,      # Shallow trees
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,   # Require more samples in leaf
                max_features='sqrt',   # Limit features per tree
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Gradient Boosting with early stopping
            'gradient_boosting_regularized': GradientBoostingClassifier(
                n_estimators=50,      # Fewer estimators
                max_depth=2,          # Very shallow trees
                learning_rate=0.01,   # Slow learning
                min_samples_split=20, # Require more samples
                min_samples_leaf=10,  # Require more samples in leaf
                subsample=0.8,        # Use only 80% of samples
                random_state=self.random_state
            )
        }
    
    def create_conservative_pipelines(self):
        """Create pipelines with conservative TF-IDF to reduce overfitting."""
        
        # Conservative TF-IDF parameters
        tfidf_params = {
            'max_features': 1000,     # Much smaller feature set
            'min_df': 5,              # Word must appear in at least 5 documents
            'max_df': 0.5,            # Ignore words in >50% of documents
            'ngram_range': (1, 1),    # Only unigrams (no bigrams)
            'sublinear_tf': True,
            'stop_words': 'english'
        }
        
        print(f"\n🛡️  CONSERVATIVE TF-IDF PARAMETERS:")
        print(f"   Max features: {tfidf_params['max_features']} (reduced from 10,000)")
        print(f"   Min document frequency: {tfidf_params['min_df']} (increased from 2)")
        print(f"   Max document frequency: {tfidf_params['max_df']} (reduced from 0.8)")
        print(f"   N-grams: {tfidf_params['ngram_range']} (unigrams only)")
        
        # Create pipelines
        for model_name, model in self.models.items():
            self.pipelines[model_name] = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('classifier', model)
            ])
    
    def train_with_validation_monitoring(self, X_train, y_train, cv_folds=5):
        """Train models while monitoring for overfitting."""
        print("\n🎯 TRAINING REGULARIZED MODELS")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, pipeline in self.pipelines.items():
            print(f"\n🔄 Training {model_name.replace('_', ' ').title()}...")
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Fit model
            pipeline.fit(X_train, y_train)
            
            # Training accuracy
            y_train_pred = pipeline.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # Store results
            self.results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            self.training_results[model_name] = train_accuracy
            
            # Calculate overfitting
            overfitting = train_accuracy - cv_scores.mean()
            
            # Status indication
            if overfitting > 0.1:
                status = "⚠️  Still Overfitting"
            elif overfitting > 0.05:
                status = "⚠️  Mild Overfitting"
            else:
                status = "✅ Good Generalization"
            
            print(f"   Training: {train_accuracy:.4f}")
            print(f"   CV Mean:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"   Overfit:  {overfitting:.4f} - {status}")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate models on test set."""
        print("\n📊 TEST SET EVALUATION")
        print("=" * 50)
        
        for model_name, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            self.test_results[model_name] = {
                'accuracy': test_accuracy,
                'predictions': y_pred
            }
            
            print(f"{model_name.replace('_', ' ').title()}: {test_accuracy:.4f}")
    
    def print_regularization_summary(self):
        """Print summary focusing on overfitting reduction."""
        print("\n" + "="*80)
        print("REGULARIZATION EFFECTIVENESS SUMMARY")
        print("="*80)
        
        print(f"{'Model':<25} {'Train':<8} {'CV':<8} {'Test':<8} {'Overfit':<8} {'Status'}")
        print("-" * 80)
        
        for model_name in self.results.keys():
            train_acc = self.training_results[model_name]
            cv_acc = self.results[model_name]['cv_mean']
            test_acc = self.test_results.get(model_name, {}).get('accuracy', 0)
            overfitting = train_acc - cv_acc
            
            if overfitting <= 0.05:
                status = "✅ FIXED"
            elif overfitting <= 0.1:
                status = "⚠️  Improved"
            else:
                status = "❌ Still High"
            
            print(f"{model_name.replace('_', ' ').title():<25} {train_acc:<8.4f} {cv_acc:<8.4f} {test_acc:<8.4f} {overfitting:<8.4f} {status}")
        
        # Best regularized model
        best_model = min(self.training_results.keys(), 
                        key=lambda x: abs(self.training_results[x] - self.results[x]['cv_mean']))
        
        print(f"\n🏆 BEST REGULARIZED MODEL: {best_model.replace('_', ' ').title()}")
        print(f"\n💡 REGULARIZATION SUCCESS FACTORS:")
        print(f"   • Reduced feature space (1K vs 10K features)")
        print(f"   • Increased min_df (5 vs 2) - removes rare words")  
        print(f"   • Strong model penalties (low C, high alpha values)")
        print(f"   • Tree constraints (shallow depth, min samples)")

# Main execution for regularized training
if __name__ == "__main__":
    print("🛡️  REGULARIZED LEGAL DOCUMENT CLASSIFIER")
    print("   Designed to Combat Overfitting")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/processed/legal_documents_preprocessed.csv')
    print(f"📂 Loaded {len(df)} documents")
    
    # Prepare data
    X = df['processed_text']
    y = df['label']
    
    # Filter rare classes
    from collections import Counter
    class_counts = Counter(y)
    classes_to_keep = [cls for cls, count in class_counts.items() if count >= 3]  # At least 3 samples
    
    mask = y.isin(classes_to_keep)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"📊 After filtering: {len(X_filtered)} documents, {len(set(y_filtered))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    
    # Initialize regularized classifier
    classifier = RegularizedLegalDocumentClassifier()
    classifier.create_conservative_pipelines()
    
    # Train with overfitting monitoring
    classifier.train_with_validation_monitoring(X_train, y_train)
    
    # Evaluate on test set
    classifier.evaluate_models(X_test, y_test)
    
    # Print regularization effectiveness
    classifier.print_regularization_summary()
    
    print(f"\n✅ REGULARIZED TRAINING COMPLETE!")
    print(f"   If overfitting persists, consider:")
    print(f"   • Data augmentation techniques")
    print(f"   • Legal-BERT with dropout")
    print(f"   • Ensemble with different regularization levels")
