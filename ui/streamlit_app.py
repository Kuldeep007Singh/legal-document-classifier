import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Load your trained model
@st.cache_resource
def load_model():
    # Adjust path to your saved model
    model_path = "models/best_models/traditional_20250901_002638.joblib"
    vectorizer_path = "models/tfidf_vectorizer.pkl"  # You'll need to save this
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

# Legal categories (adjust based on your model)
LEGAL_CATEGORIES = [
    'Contracts & Agreements',
    'EU Regulations', 
    'Human Rights Cases',
    'Supreme Court Cases',
    'Terms of Service'
]

def predict_document(text, model, vectorizer):
    """Predict the legal category of input text."""
    # Transform text using the same vectorizer from training
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Get top 3 predictions with confidence
    prob_dict = {LEGAL_CATEGORIES[i]: prob for i, prob in enumerate(probabilities)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return prediction, sorted_probs

def main():
    st.set_page_config(
        page_title="Legal Document Classifier", 
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Classifier")
    st.markdown("**Classify legal documents into 5 broad categories with 94.85% accuracy**")
    
    # Load model
    try:
        model, vectorizer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Enter Legal Document Text")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload File"]
        )
        
        user_text = ""
        
        if input_method == "Type/Paste Text":
            user_text = st.text_area(
                "Enter your legal document text here:",
                height=300,
                placeholder="Paste your legal document text here... (contracts, regulations, court cases, etc.)"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file", 
                type=['txt', 'pdf'],
                help="Upload a .txt or .pdf file containing your legal document"
            )
            
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    user_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("PDF support coming soon! Please use .txt files for now.")
        
        # Predict button
        if st.button("🔍 Classify Document", type="primary"):
            if user_text.strip():
                with st.spinner("🔄 Analyzing document..."):
                    try:
                        prediction, probabilities = predict_document(user_text, model, vectorizer)
                        
                        # Show results in col2
                        with col2:
                            st.subheader("📊 Classification Results")
                            
                            # Main prediction
                            st.success(f"**Primary Category:**\n🎯 {prediction}")
                            
                            # Confidence scores
                            st.subheader("📈 Confidence Scores")
                            for category, prob in probabilities[:3]:
                                confidence = prob * 100
                                st.metric(
                                    label=category,
                                    value=f"{confidence:.1f}%",
                                    delta=None
                                )
                                # Progress bar for visual representation
                                st.progress(prob)
                            
                            # Document stats
                            st.subheader("📋 Document Stats")
                            st.info(f"""
                            **Word Count:** {len(user_text.split())}
                            **Character Count:** {len(user_text)}
                            **Predicted Category:** {prediction}
                            **Confidence:** {probabilities[0][1]*100:.1f}%
                            """)
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text to classify!")
    
    with col2:
        if not st.session_state.get('prediction_made', False):
            st.subheader("ℹ️ About This Classifier")
            st.info("""
            **Model Performance:**
            - ✅ 94.85% Accuracy
            - 🔄 5-fold Cross-Validation 
            - 📊 Trained on 184,000+ legal documents
            
            **Categories:**
            - Contracts & Agreements
            - EU Regulations  
            - Human Rights Cases
            - Supreme Court Cases
            - Terms of Service
            
            **How to Use:**
            1. Paste your legal document text
            2. Click "Classify Document"  
            3. View prediction and confidence scores
            """)

if __name__ == "__main__":
    main()
