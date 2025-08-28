import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class LegalDocumentBalancer:
    """
    Balance legal document dataset using various techniques.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    def analyze_imbalance(self, y):
        """Analyze class imbalance in the dataset."""
        class_counts = Counter(y)
        total = len(y)
        
        print("Class Distribution:")
        for label, count in class_counts.most_common():
            percentage = (count / total) * 100
            print(f"{label}: {count} documents ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        ratio = max_count / min_count
        
        print(f"\nImbalance Ratio: {ratio:.2f}:1")
        return class_counts, ratio
    
    def random_oversample(self, X, y):
        """Random oversampling of minority classes."""
        print("\nApplying Random Oversampling...")
        
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)
        
        # Convert back to original format
        X_resampled = pd.Series([x for x in X_resampled])
        y_resampled = pd.Series(y_resampled)
        
        print(f"Original size: {len(X)}")
        print(f"Resampled size: {len(X_resampled)}")
        
        return X_resampled, y_resampled
    
    def random_undersample(self, X, y):
        """Random undersampling of majority classes."""
        print("\nApplying Random Undersampling...")
        
        rus = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X.values.reshape(-1, 1), y)
        
        # Convert back to original format
        X_resampled = pd.Series([x for x in X_resampled])
        y_resampled = pd.Series(y_resampled)
        
        print(f"Original size: {len(X)}")
        print(f"Resampled size: {len(X_resampled)}")
        
        return X_resampled, y_resampled
    
    def smote_oversample(self, X, y, k_neighbors=3):
        """SMOTE oversampling for text data."""
        print("\nApplying SMOTE Oversampling...")
        
        # Convert text to numerical features for SMOTE
        X_vectorized = self.vectorizer.fit_transform(X).toarray()
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state, k_neighbors=min(k_neighbors, 
                     Counter(y)[min(Counter(y), key=Counter(y).get)] - 1))
        X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)
        
        print(f"Original size: {len(X)}")
        print(f"Resampled size: {len(X_resampled)}")
        
        # Note: X_resampled is now vectorized, would need inverse transform for text
        # For practical purposes, we'll use class weights instead
        return None, None  # Return None to indicate use class weights
    
    def balanced_class_weights(self, y):
        """Calculate class weights for balanced learning."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        print("\nCalculated Class Weights:")
        for class_label, weight in weight_dict.items():
            print(f"{class_label}: {weight:.3f}")
        
        return weight_dict
    
    def stratified_sample_balance(self, df, text_col='processed_text', label_col='label', 
                                samples_per_class=50):
        """Create a stratified balanced sample."""
        print(f"\nCreating balanced sample with {samples_per_class} samples per class...")
        
        balanced_dfs = []
        
        for class_label in df[label_col].unique():
            class_df = df[df[label_col] == class_label]
            
            if len(class_df) >= samples_per_class:
                # Sample if we have enough
                sampled = class_df.sample(n=samples_per_class, random_state=self.random_state)
            else:
                # Take all if we don't have enough
                sampled = class_df
                print(f"Warning: {class_label} has only {len(class_df)} samples")
            
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"Balanced dataset size: {len(balanced_df)}")
        print("New class distribution:")
        print(balanced_df[label_col].value_counts())
        
        return balanced_df

# Usage example
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/processed/legal_documents_preprocessed.csv')
    
    # Initialize balancer
    balancer = LegalDocumentBalancer()
    
    # Analyze current imbalance
    class_counts, ratio = balancer.analyze_imbalance(df['label'])
    
    if ratio > 3:  # Significant imbalance
        print("\n🎯 APPLYING DATA BALANCING...")
        
        # Option 1: Create balanced sample
        balanced_df = balancer.stratified_sample_balance(df, samples_per_class=30)
        
        # Option 2: Calculate class weights for model training
        class_weights = balancer.balanced_class_weights(df['label'])
        
        # Save balanced dataset
        balanced_df.to_csv('data/processed/legal_documents_balanced.csv', index=False)
        print("Balanced dataset saved!")
        
        # Save class weights for model training
        import json
        with open('data/processed/class_weights.json', 'w') as f:
            json.dump(class_weights, f)
        print("Class weights saved!")
    
    else:
        print("✅ Dataset is already relatively balanced")
