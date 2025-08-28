import re
import string
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class LegalTextPreprocessor:
    """
    Preprocessor for legal document text with domain-specific cleaning.
    """
    
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_requirements()
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()
        
        # Legal-specific stopwords (extend default stopwords)
        self.legal_stopwords = set(stopwords.words('english')).union({
            'agreement', 'contract', 'party', 'parties', 'shall', 'may', 'will',
            'hereby', 'herein', 'thereof', 'therein', 'whereof', 'whereas',
            'subject', 'section', 'clause', 'provision', 'said', 'such',
            'including', 'pursuant', 'accordance', 'respect', 'term', 'terms',
            'condition', 'conditions', 'applicable', 'effective', 'date'
        })
    
    def _download_nltk_requirements(self):
        """Download required NLTK data silently."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK requirements...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text extracted from PDFs.
        
        :param text: Raw text from PDF extraction
        :return: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'\bpage\s+\d+\b', '', text)
        text = re.sub(r'\d+\s*of\s*\d+', '', text)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation while keeping sentence structure
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\-{2,}', '-', text)
        
        # Remove standalone numbers (often page refs, amounts without context)
        text = re.sub(r'\b\d+\.?\d*\b', ' ', text)
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text and apply lemmatization.
        
        :param text: Cleaned text
        :param remove_stopwords: Whether to remove stopwords
        :return: List of processed tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip if too short or is punctuation
            if len(token) < 3 or token in string.punctuation:
                continue
                
            # Skip stopwords if requested
            if remove_stopwords and token.lower() in self.legal_stopwords:
                continue
            
            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def preprocess_text(self, text: str, join_tokens: bool = True) -> str or List[str]:
        """
        Complete text preprocessing pipeline.
        
        :param text: Raw text
        :param join_tokens: Whether to return joined string or token list
        :return: Processed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        if join_tokens:
            return ' '.join(tokens)
        else:
            return tokens
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                          label_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess entire dataset.
        
        :param df: Input DataFrame
        :param text_column: Name of text column
        :param label_column: Name of label column
        :return: DataFrame with preprocessed text and encoded labels
        """
        print("Preprocessing dataset...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Preprocess text
        print("Cleaning and preprocessing text...")
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: self.preprocess_text(x, join_tokens=True)
        )
        
        # Filter out empty processed texts
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        final_count = len(processed_df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} documents with empty processed text")
        
        # Add text statistics
        processed_df['processed_word_count'] = processed_df['processed_text'].apply(
            lambda x: len(x.split()) if x else 0
        )
        processed_df['processed_char_count'] = processed_df['processed_text'].apply(len)
        
        # Encode labels
        processed_df['label_encoded'] = self.label_encoder.fit_transform(processed_df[label_column])
        
        print(f"Preprocessing complete!")
        print(f"Final dataset size: {len(processed_df)} documents")
        print(f"Average processed word count: {processed_df['processed_word_count'].mean():.1f}")
        
        return processed_df
    
    def get_label_mapping(self) -> dict:
        """Get mapping from encoded labels to original labels."""
        if hasattr(self.label_encoder, 'classes_'):
            return {i: label for i, label in enumerate(self.label_encoder.classes_)}
        return {}

if __name__ == "__main__":
    # Test the preprocessor
    # Load your processed dataset
    df = pd.read_csv('data/processed/legal_documents_sample.csv')
    
    # Initialize preprocessor
    preprocessor = LegalTextPreprocessor()
    
    # Preprocess the dataset
    processed_df = preprocessor.preprocess_dataset(df)
    
    # Display results
    print(f"\nPreprocessing Results:")
    print(f"Original dataset shape: {df.shape}")
    print(f"Processed dataset shape: {processed_df.shape}")
    
    print(f"\nLabel mapping:")
    label_mapping = preprocessor.get_label_mapping()
    for encoded, original in label_mapping.items():
        print(f"  {encoded}: {original}")
    
    print(f"\nSample processed texts:")
    for i in range(min(3, len(processed_df))):
        original_text = processed_df.iloc[i]['text'][:200]
        processed_text = processed_df.iloc[i]['processed_text'][:200]
        print(f"\nDocument {i+1} ({processed_df.iloc[i]['label']}):")
        print(f"Original: {original_text}...")
        print(f"Processed: {processed_text}...")
    
    # Save processed dataset
    output_path = 'data/processed/legal_documents_preprocessed.csv'
    processed_df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")
