# ui/utils.py
import joblib
import os
from typing import Tuple, List

def load_model_and_vectorizer(model_dir: str = "models/best_models"):
    """Load the trained model and vectorizer."""
    # Find the latest model file
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') and f.startswith('traditional')]
    
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = sorted(model_files)[-1]  # Get the most recent
    model_path = os.path.join(model_dir, latest_model)
    
    # Load vectorizer (you'll need to save this during training)
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

LEGAL_CATEGORIES = [
    'Contracts & Agreements',
    'EU Regulations', 
    'Human Rights Cases',
    'Supreme Court Cases',
    'Terms of Service'
]
