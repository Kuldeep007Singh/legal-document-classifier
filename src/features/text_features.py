# Code Path: src/features/text_features.py

import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

# Configure logging to show output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Optional imports with fallback
try:
    from gensim.models import Word2Vec, FastText
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
    print("✓ Gensim is available")
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None
    FastText = None
    Doc2Vec = None
    TaggedDocument = None
    print("✗ Gensim is not available - Word2Vec and Doc2Vec features disabled")

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract various text features for legal documents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fitted_extractors = {}
        print("TextFeatureExtractor initialized")
        
    def extract_tfidf_features(self, 
                              texts: List[str],
                              max_features: int = 1000,  # Reduced for demo
                              ngram_range: Tuple[int, int] = (1, 2),
                              min_df: int = 1,
                              max_df: float = 0.95,
                              fit: bool = True) -> np.ndarray:
        """Extract TF-IDF features"""
        
        print(f"Extracting TF-IDF features from {len(texts)} texts...")
        
        if fit or 'tfidf' not in self.fitted_extractors:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
            
            features = vectorizer.fit_transform(texts)
            self.fitted_extractors['tfidf'] = vectorizer
            
            print(f"✓ TF-IDF features extracted: {features.shape}")
        else:
            vectorizer = self.fitted_extractors['tfidf']
            features = vectorizer.transform(texts)
            print(f"✓ TF-IDF features transformed: {features.shape}")
        
        return features.toarray()

    def extract_count_features(self,
                              texts: List[str],
                              max_features: int = 1000,
                              ngram_range: Tuple[int, int] = (1, 2),
                              fit: bool = True) -> np.ndarray:
        """Extract count-based features"""
        
        print(f"Extracting count features from {len(texts)} texts...")
        
        if fit or 'count' not in self.fitted_extractors:
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )
            
            features = vectorizer.fit_transform(texts)
            self.fitted_extractors['count'] = vectorizer
            print(f"✓ Count features extracted: {features.shape}")
        else:
            vectorizer = self.fitted_extractors['count']
            features = vectorizer.transform(texts)
            print(f"✓ Count features transformed: {features.shape}")
        
        return features.toarray()


class LegalFeatureExtractor:
    """Extract legal document-specific features"""
    
    def __init__(self):
        self.legal_patterns = self._init_legal_patterns()
        self.text_extractor = TextFeatureExtractor()
        print("LegalFeatureExtractor initialized")
    
    def _init_legal_patterns(self) -> Dict[str, str]:
        """Initialize legal document patterns"""
        return {
            'citations': r'\b\d+\s+[A-Z][a-z]*\.?\s+\d+\b|\b\d+\s+U\.S\.C?\.\s*§?\s*\d+\b',
            'statutes': r'\b\d+\s+U\.S\.C\.?\s*§\s*\d+(?:\([a-z]\))?\b',
            'case_citations': r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+\b',
            'monetary_amounts': r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b',
            'percentages': r'\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'legal_entities': r'\b(?:LLC|Inc\.?|Corp\.?|Ltd\.?|LP|LLP|Partnership|Corporation|Company)\b',
            'contract_sections': r'\b(?:Article|Section|Clause|Paragraph|Subsection)\s+\d+(?:\.\d+)*\b',
            'legal_terms': r'\b(?:whereas|therefore|hereby|herein|hereof|hereunder|aforementioned|aforesaid)\b'
        }
    
    def extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract features based on legal patterns"""
        features = {}
        word_count = len(text.split())
        
        for pattern_name, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'{pattern_name}_count'] = len(matches)
            features[f'{pattern_name}_density'] = len(matches) / max(word_count, 1)
        
        return features
    
    def extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract document structure features"""
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Average lengths
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
        else:
            features['avg_word_length'] = 0
        
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences if sent.strip()])
        else:
            features['avg_sentence_length'] = 0
        
        # Complexity measures
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        features['punctuation_density'] = sum(1 for char in text if char in string.punctuation) / max(len(text), 1)
        
        return features
    
    def extract_legal_complexity_features(self, text: str) -> Dict[str, Any]:
        """Extract features indicating legal document complexity"""
        features = {}
        word_count = len(text.split())
        
        # Modal verbs (indicating obligations)
        modal_verbs = ['shall', 'must', 'may', 'will', 'should', 'could', 'would', 'might']
        modal_count = sum(len(re.findall(rf'\b{verb}\b', text, re.IGNORECASE)) for verb in modal_verbs)
        features['modal_verb_density'] = modal_count / max(word_count, 1)
        
        # Passive voice indicators
        passive_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w*ed\b',
            r'\b(?:is|are|was|were|been|being)\s+\w*en\b'
        ]
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_patterns)
        features['passive_voice_density'] = passive_count / max(word_count, 1)
        
        # Legal connectors
        legal_connectors = ['whereas', 'therefore', 'furthermore', 'notwithstanding']
        connector_count = sum(len(re.findall(rf'\b{connector}\b', text, re.IGNORECASE)) for connector in legal_connectors)
        features['legal_connector_density'] = connector_count / max(word_count, 1)
        
        return features


if __name__ == "__main__":
    print("="*50)
    print("LEGAL DOCUMENT FEATURE EXTRACTION DEMO")
    print("="*50)
    
    # Example usage with more detailed sample texts
    sample_texts = [
        "This agreement is entered into on January 15, 2024 between Company Inc. and Client LLC. The parties agree to the following terms and conditions whereas the first party shall provide services.",
        "The party of the first part shall provide consulting services as outlined herein. Payment of $10,000 is due upon completion. This contract may be terminated with 30 days notice.",
        "Termination of this contract may occur upon thirty (30) days written notice. The Company shall indemnify and hold harmless the Client from any damages exceeding 5% of the total contract value."
    ]
    
    print(f"\nProcessing {len(sample_texts)} sample legal texts...")
    
    try:
        # Legal feature extraction
        print("\n1. Initializing Legal Feature Extractor...")
        legal_extractor = LegalFeatureExtractor()
        
        # Text feature extraction
        print("\n2. Initializing Text Feature Extractor...")
        text_extractor = TextFeatureExtractor()
        
        print("\n3. Extracting TF-IDF features...")
        tfidf_features = text_extractor.extract_tfidf_features(sample_texts)
        print(f"   TF-IDF features shape: {tfidf_features.shape}")
        
        print("\n4. Extracting Count features...")
        count_features = text_extractor.extract_count_features(sample_texts)
        print(f"   Count features shape: {count_features.shape}")
        
        print("\n5. Extracting Legal Pattern features...")
        for i, text in enumerate(sample_texts):
            pattern_features = legal_extractor.extract_pattern_features(text)
            print(f"   Text {i+1} pattern features:")
            for key, value in pattern_features.items():
                if value > 0:  # Only show non-zero features
                    print(f"     {key}: {value}")
        
        print("\n6. Extracting Structural features...")
        for i, text in enumerate(sample_texts):
            structural_features = legal_extractor.extract_structural_features(text)
            print(f"   Text {i+1} structural features:")
            for key, value in structural_features.items():
                print(f"     {key}: {value:.2f}")
        
        print("\n7. Extracting Legal Complexity features...")
        for i, text in enumerate(sample_texts):
            complexity_features = legal_extractor.extract_legal_complexity_features(text)
            print(f"   Text {i+1} complexity features:")
            for key, value in complexity_features.items():
                if value > 0:  # Only show non-zero features
                    print(f"     {key}: {value:.4f}")
        
        print("\n" + "="*50)
        print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
