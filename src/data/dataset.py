# Code Path: src/data/dataset.py
import sys
import os
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent.parent  # adjust number of parents as needed
sys.path.insert(0, str(project_root))

# Now import settings
from project_config.settings import settings

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
import logging
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class LegalDocumentDataset(Dataset):
    """PyTorch Dataset for legal document classification"""
    
    def __init__(self,
                 texts: List[str],
                 labels: List[str],
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 label_encoder: Optional[LabelEncoder] = None):
        
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Handle labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = label_encoder.transform(labels)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Dataset created with {len(texts)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.encoded_labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from encoded labels to original labels"""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
    
    def save_label_encoder(self, path: Path):
        """Save label encoder for later use"""
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to {path}")
    
    @classmethod
    def load_label_encoder(cls, path: Path) -> LabelEncoder:
        """Load saved label encoder"""
        with open(path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Label encoder loaded from {path}")
        return label_encoder

class MultiDatasetLoader:
    """Loader for multiple legal datasets"""
    
    def __init__(self, 
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.datasets = {}
        self.label_encoders = {}
    
    def load_dataset_from_csv(self, 
                            csv_path: Path,
                            dataset_name: str,
                            text_column: str = 'text',
                            label_column: str = 'label') -> LegalDocumentDataset:
        """Load dataset from CSV file"""
        logger.info(f"Loading dataset {dataset_name} from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if text_column not in df.columns:
            # Try to find text column with different names
            text_candidates = ['text', 'document', 'content', 'description', 'case_text']
            text_column = next((col for col in text_candidates if col in df.columns), None)
            
            if text_column is None:
                raise ValueError(f"No text column found in {csv_path}")
        
        if label_column not in df.columns:
            # Try to find label column with different names
            label_candidates = ['label', 'category', 'class', 'type', 'classification']
            label_column = next((col for col in label_candidates if col in df.columns), None)
            
            if label_column is None:
                raise ValueError(f"No label column found in {csv_path}")
        
        # Clean data
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].str.strip() != '']
        
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()
        
        # Create or reuse label encoder
        if dataset_name in self.label_encoders:
            label_encoder = self.label_encoders[dataset_name]
        else:
            label_encoder = None
        
        dataset = LegalDocumentDataset(
            texts=texts,
            labels=labels,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            label_encoder=label_encoder
        )
        
        # Store label encoder
        self.label_encoders[dataset_name] = dataset.label_encoder
        self.datasets[dataset_name] = dataset
        
        return dataset
    
    def create_dataloaders(self, 
                          dataset: LegalDocumentDataset,
                          batch_size: int = 16,
                          shuffle: bool = True,
                          num_workers: int = 2) -> DataLoader:
        """Create DataLoader from dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def load_all_splits(self, 
                       base_path: Path,
                       dataset_name: str,
                       text_column: str = 'text',
                       label_column: str = 'label') -> Dict[str, LegalDocumentDataset]:
        """Load train, validation, and test splits for a dataset"""
        
        splits = {}
        split_files = {
            'train': f"{dataset_name}_train.csv",
            'validation': f"{dataset_name}_validation.csv", 
            'test': f"{dataset_name}_test.csv"
        }
        
        # Load training data first to create label encoder
        train_path = base_path / split_files['train']
        if train_path.exists():
            splits['train'] = self.load_dataset_from_csv(
                train_path, dataset_name, text_column, label_column
            )
        
        # Load validation and test with same label encoder
        for split_name, filename in split_files.items():
            if split_name == 'train':  # Already loaded
                continue
                
            file_path = base_path / filename
            if file_path.exists():
                splits[split_name] = self.load_dataset_from_csv(
                    file_path, f"{dataset_name}_{split_name}", text_column, label_column
                )
                # Use same label encoder as training
                splits[split_name].label_encoder = splits['train'].label_encoder
                splits[split_name].encoded_labels = splits['train'].label_encoder.transform(
                    splits[split_name].labels
                )
        
        return splits
    
    def get_class_weights(self, dataset: LegalDocumentDataset) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(dataset.encoded_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=dataset.encoded_labels
        )
        
        return torch.FloatTensor(class_weights)

class DataCollator:
    """Custom data collator for legal documents"""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Utility functions
def create_stratified_split(df: pd.DataFrame, 
                          label_column: str = 'label',
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Create stratified train/validation/test splits"""
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[label_column],
        random_state=random_state
    )
    
    return {
        'train': train,
        'validation': val,
        'test': test
    }

if __name__ == "__main__":
    # Example usage
    
    
    # Initialize loader
    loader = MultiDatasetLoader()
    
    # Load SCOTUS dataset
    base_path = Path("data/raw")
    datasets = loader.load_all_splits(base_path, "scotus")
    
    print(f"Loaded splits: {list(datasets.keys())}")
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {len(dataset)} samples, {dataset.num_classes} classes")
    
    # Create DataLoader
    if 'train' in datasets:
        train_loader = loader.create_dataloaders(datasets['train'], batch_size=8)
        
        # Test batch
        for batch in train_loader:
            print("Batch keys:", batch.keys())
            print("Input IDs shape:", batch['input_ids'].shape)
            print("Labels shape:", batch['labels'].shape)
            break