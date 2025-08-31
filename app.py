# At the top of your existing app.py, update the model loading:

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

# Update model path to use LEDGAR model
MODEL_PATH = "models/ledgar_legal_bert"  # Changed from old model path

@st.cache_resource
def load_model():
    """Load the trained LEDGAR Legal-BERT model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Rest of your Streamlit app remains the same...
