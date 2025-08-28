import pandas as pd
from typing import List, Dict, Optional
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Use the correct class name from your data_loader.py file
from src.data.data_loader import PDFFolderDataLoader  # Changed from RecursivePDFFolderDataLoader
from src.data.document_parser import PDFTextExtractor

class LegalDocumentDataset:
    """
    Combines data loading and text extraction to create a labeled dataset.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.loader = PDFFolderDataLoader(data_dir)  # Updated class name
        self.extractor = PDFTextExtractor()
        
    def create_dataset(self, max_docs: Optional[int] = None) -> pd.DataFrame:
        """
        Create a DataFrame with extracted text and labels.
        
        :param max_docs: Maximum number of documents to process (None for all)
        :return: DataFrame with 'text' and 'label' columns
        """
        # Load all PDF file paths and labels
        print("Loading PDF file paths...")
        docs = self.loader.load_filepaths_and_labels()
        
        if not docs:
            print("No documents found!")
            return pd.DataFrame()
        
        print(f"Found {len(docs)} documents.")
        
        # Limit number of documents if specified
        if max_docs:
            docs = docs[:max_docs]
            print(f"Processing first {len(docs)} documents.")
        
        # Extract text from each PDF
        dataset = []
        failed_extractions = 0
        
        for i, doc in enumerate(docs):
            print(f"Processing {i+1}/{len(docs)}: {os.path.basename(doc['filepath'])}")
            
            text = self.extractor.extract_text_from_pdf(doc['filepath'])
            
            if text and len(text.strip()) > 0:
                dataset.append({
                    'filepath': doc['filepath'],
                    'text': text,
                    'label': doc['label'],
                    'filename': os.path.basename(doc['filepath'])
                })
            else:
                failed_extractions += 1
                print(f"Failed to extract text from: {doc['filepath']}")
        
        print(f"\nDataset creation complete!")
        print(f"Successfully processed: {len(dataset)} documents")
        print(f"Failed extractions: {failed_extractions}")
        
        # Create DataFrame
        df = pd.DataFrame(dataset)
        return df

if __name__ == "__main__":
    # Update this path to your actual CUAD data directory
    data_dir = r"C:\Users\Lenovo\Downloads\CUAD_v11\CUAD_v1\full_contract_pdf"
    
    # Create dataset (limit to first 10 documents for testing)
    dataset_creator = LegalDocumentDataset(data_dir)
    df = dataset_creator.create_dataset(max_docs=None)
    
    if not df.empty:
        print(f"\nDataset shape: {df.shape}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        print(f"\nFirst few rows:")
        print(df[['filename', 'label']].head())
        
        # Save to CSV for later use
        output_path = "data/processed/legal_documents_sample.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")
