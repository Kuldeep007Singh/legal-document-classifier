# Code Path: src/data/preprocessor.py
import numpy as np
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class LegalTextPreprocessor:
    """Comprehensive text preprocessing for legal documents"""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 lowercase: bool = True,
                 min_token_length: int = 2,
                 max_token_length: int = 100):
        
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Initialize tools
        self._init_nltk_tools()
        self._init_spacy_model()
        self._init_legal_patterns()
    
    def _init_nltk_tools(self):
        """Initialize NLTK tools"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add legal-specific stopwords
        legal_stopwords = {
            'shall', 'may', 'must', 'will', 'would', 'could', 'should',
            'party', 'parties', 'agreement', 'contract', 'document',
            'section', 'subsection', 'paragraph', 'clause', 'provision',
            'herein', 'hereby', 'hereof', 'hereunder', 'heretofore',
            'whereas', 'therefore', 'furthermore', 'notwithstanding'
        }
        
        # Option to keep legal stopwords for legal document classification
        if not hasattr(self, 'keep_legal_terms'):
            self.keep_legal_terms = True
        
        if not self.keep_legal_terms:
            self.stop_words.update(legal_stopwords)
    
    def _init_spacy_model(self):
        """Initialize spaCy model for advanced NLP"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _init_legal_patterns(self):
        """Initialize legal document patterns"""
        self.legal_patterns = {
            'citations': r'\b\d+\s+[A-Z][a-z]*\.?\s+\d+\b|\b\d+\s+U\.S\.C?\.\s*§?\s*\d+\b',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
            'monetary': r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b',
            'percentages': r'\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b',
            'legal_entities': r'\b(?:LLC|Inc\.?|Corp\.?|Ltd\.?|LP|LLP|Partnership|Corporation|Company)\b',
            'contract_terms': r'\b(?:Term|Duration|Period):\s*\d+\s*(?:days?|months?|years?)\b'
        }
    
    def preprocess_text(self, text: str, preserve_legal_structure: bool = True) -> str:
        """
        Main preprocessing function
        
        Args:
            text: Input text to preprocess
            preserve_legal_structure: Whether to preserve legal document structure
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Store original for structure preservation
        original_text = text
        
        # Basic cleaning
        text = self._basic_cleaning(text)
        
        # Legal-specific preprocessing
        if preserve_legal_structure:
            text = self._preserve_legal_structure(text)
        
        # Advanced preprocessing
        text = self._advanced_preprocessing(text)
        
        return text.strip()
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve legal symbols
        if not self.remove_punctuation:
            # Keep important punctuation for legal texts
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'§]', ' ', text)
        
        # Handle line breaks in legal documents
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text
    
    def _preserve_legal_structure(self, text: str) -> str:
        """Preserve important legal document structure"""
        # Preserve section numbering
        text = re.sub(r'\n(\d+\.?\d*\.?\s)', r' SECTION_\1', text)
        
        # Preserve legal citations
        citation_pattern = self.legal_patterns['citations']
        citations = re.findall(citation_pattern, text)
        for i, citation in enumerate(citations):
            text = text.replace(citation, f' LEGAL_CITATION_{i} ')
        
        # Preserve monetary amounts
        monetary_pattern = self.legal_patterns['monetary']
        amounts = re.findall(monetary_pattern, text)
        for i, amount in enumerate(amounts):
            text = text.replace(amount, f' MONETARY_AMOUNT_{i} ')
        
        # Preserve dates
        date_pattern = self.legal_patterns['dates']
        dates = re.findall(date_pattern, text)
        for i, date in enumerate(dates):
            text = text.replace(date, f' LEGAL_DATE_{i} ')
        
        return text
    
    def _advanced_preprocessing(self, text: str) -> str:
        """Advanced preprocessing using spaCy and NLTK"""
        if self.lowercase:
            text = text.lower()
        
        # Tokenization and lemmatization using spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                # Skip if token meets removal criteria
                if (self.remove_stopwords and token.text.lower() in self.stop_words or
                    self.remove_numbers and token.like_num or
                    self.remove_punctuation and token.is_punct or
                    len(token.text) < self.min_token_length or
                    len(token.text) > self.max_token_length):
                    continue
                
                # Apply lemmatization
                if self.lemmatize and not token.like_num:
                    token_text = token.lemma_
                else:
                    token_text = token.text
                
                tokens.append(token_text)
            
            text = ' '.join(tokens)
        else:
            # Fallback to NLTK processing
            tokens = nltk.word_tokenize(text)
            
            if self.remove_stopwords:
                tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
            if self.remove_punctuation:
                tokens = [token for token in tokens if token not in string.punctuation]
            
            if self.remove_numbers:
                tokens = [token for token in tokens if not token.isdigit()]
            
            # Filter by length
            tokens = [token for token in tokens 
                     if self.min_token_length <= len(token) <= self.max_token_length]
            
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            text = ' '.join(tokens)
        
        return text
    
    def extract_legal_features(self, text: str) -> Dict[str, Any]:
        """Extract legal document-specific features"""
        features = {}
        
        # Count legal patterns
        for pattern_name, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'{pattern_name}_count'] = len(matches)
            features[f'{pattern_name}_density'] = len(matches) / max(len(text.split()), 1)
        
        # Document structure features
        features['num_sentences'] = len(nltk.sent_tokenize(text))
        features['num_paragraphs'] = len([p for p in text.split('\n\n') if p.strip()])
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in nltk.sent_tokenize(text)]) if text else 0
        
        # Legal complexity indicators
        features['passive_voice_count'] = len(re.findall(r'\b(?:is|are|was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE))
        features['modal_verb_count'] = len(re.findall(r'\b(?:shall|must|may|will|should|could|would)\b', text, re.IGNORECASE))
        features['legal_connector_count'] = len(re.findall(r'\b(?:whereas|therefore|furthermore|notwithstanding|provided that)\b', text, re.IGNORECASE))
        
        return features
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          text_column: str = 'text',
                          output_path: Optional[Path] = None) -> pd.DataFrame:
        """Preprocess an entire dataset"""
        logger.info(f"Preprocessing dataset with {len(df)} samples")
        
        processed_df = df.copy()
        
        # Apply preprocessing to text column
        processed_df[f'{text_column}_processed'] = processed_df[text_column].apply(
            lambda x: self.preprocess_text(str(x)) if pd.notna(x) else ""
        )
        
        # Extract legal features
        legal_features = processed_df[text_column].apply(self.extract_legal_features)
        feature_df = pd.DataFrame(legal_features.tolist())
        
        # Combine with original data
        processed_df = pd.concat([processed_df, feature_df], axis=1)
        
        # Remove rows with empty processed text
        processed_df = processed_df[processed_df[f'{text_column}_processed'].str.len() > 0]
        
        logger.info(f"Preprocessing complete. {len(processed_df)} samples remaining")
        
        if output_path:
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Saved preprocessed dataset to {output_path}")
        
        return processed_df
    
    def batch_preprocess(self, input_dir: Path, output_dir: Path, 
                        file_pattern: str = "*.csv") -> None:
        """Batch preprocess all CSV files in a directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        csv_files = list(input_dir.glob(file_pattern))
        
        for csv_file in csv_files:
            logger.info(f"Processing file: {csv_file}")
            
            try:
                df = pd.read_csv(csv_file)
                processed_df = self.preprocess_dataset(df)
                
                output_file = output_dir / f"{csv_file.stem}_processed.csv"
                processed_df.to_csv(output_file, index=False)
                
                logger.info(f"Saved processed file: {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                continue

# Custom tokenizer for legal documents
class LegalTokenizer:
    """Specialized tokenizer for legal documents"""
    
    def __init__(self):
        self.legal_abbreviations = {
            'corp.': 'corporation',
            'inc.': 'incorporated',
            'ltd.': 'limited',
            'llc': 'limited liability company',
            'u.s.c.': 'united states code',
            'cfr': 'code of federal regulations',
            'et al.': 'and others',
            'v.': 'versus',
            'vs.': 'versus'
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common legal abbreviations"""
        for abbrev, expansion in self.legal_abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def tokenize_legal_text(self, text: str) -> List[str]:
        """Tokenize text while preserving legal terms"""
        # Expand abbreviations first
        text = self.expand_abbreviations(text)
        
        # Preserve legal citations and references
        text = re.sub(r'(\d+\s+U\.S\.C?\.?\s*§?\s*\d+)', r'LEGAL_CITATION_\1', text)
        
        # Standard tokenization
        tokens = nltk.word_tokenize(text)
        
        return tokens

if __name__ == "__main__":
    # Example usage
    preprocessor = LegalTextPreprocessor()
    
    sample_text = """
    This AGREEMENT is entered into on January 1, 2024, between Company Inc. 
    and Client LLC. The parties agree to the following terms and conditions...
    """
    
    processed_text = preprocessor.preprocess_text(sample_text)
    print("Original:", sample_text)
    print("Processed:", processed_text)
    
    # Extract features
    features = preprocessor.extract_legal_features(sample_text)
    print("Legal features:", features)