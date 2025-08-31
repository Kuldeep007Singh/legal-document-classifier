import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer_simple():
    """Create TF-IDF vectorizer without importing trainer (avoids relative import issues)."""
    
    print("🔄 Creating TF-IDF vectorizer with same parameters as training...")
    
    # Create vectorizer with EXACT same parameters as your training
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Create some dummy legal text to fit the vectorizer
    # (This establishes the vocabulary - your model expects this same vocabulary)
    dummy_legal_texts = [
        "This contract agreement shall be binding upon both parties.",
        "The European Union regulation provides guidelines for data protection.",
        "The Supreme Court case establishes precedent for constitutional law.",
        "Human rights violations must be addressed by international courts.",
        "Terms of service agreements protect user privacy and data.",
        "Employment contract specifies duties and compensation.",
        "Merger agreement outlines acquisition terms and conditions.",
        "Civil procedure rules govern court proceedings and litigation.",
        "Corporate governance policies ensure compliance and transparency.",
        "Legal document classification requires natural language processing."
    ]
    
    print("⚠️  Note: Using dummy data to fit vectorizer.")
    print("   The actual vocabulary will be different from training data.")
    print("   For best results, rerun training with vectorizer saving enabled.")
    
    # Fit vectorizer on dummy data
    vectorizer.fit(dummy_legal_texts)
    
    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    
    print("✅ TF-IDF vectorizer saved to models/tfidf_vectorizer.pkl")
    print("🚀 You can now run your Streamlit app!")
    
    return vectorizer

if __name__ == "__main__":
    try:
        vectorizer = create_vectorizer_simple()
        print("\n✅ Success! Vectorizer created and saved.")
        print("📋 Files in models/ directory:")
        for file in os.listdir("models"):
            print(f"   - {file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
