# Code Path: src/data/data_loader.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from project_config import settings

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from project_config.settings import settings

logger = logging.getLogger(__name__)


class LegalDatasetLoader:
    """Loads and manages legal document classification datasets"""
    
    def __init__(self, config=None):
        self.config = config or settings
        self.datasets = {}
        self.label_encoders = {}
        
    def load_dataset(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load a specific dataset (train, validation, test splits)
        
        Args:
            dataset_name: Name of the dataset (scotus, ledgar, eurlex, etc.)
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            dataset_splits = {}
            
            for split in ['train', 'validation', 'test']:
                file_path = self.config.get_dataset_path(dataset_name, split)
                
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {split} split: {len(df)} samples")
                    dataset_splits[split] = df
                else:
                    logger.warning(f"File not found: {file_path}")
            
            if not dataset_splits:
                raise FileNotFoundError(f"No data files found for dataset: {dataset_name}")
            
            # Store the dataset
            self.datasets[dataset_name] = dataset_splits
            
            return dataset_splits
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def load_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available datasets"""
        logger.info("Loading all datasets")
        
        all_datasets = {}
        
        for dataset_name in self.config.data.datasets.keys():
            try:
                dataset = self.load_dataset(dataset_name)
                all_datasets[dataset_name] = dataset
                logger.info(f"Successfully loaded {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {str(e)}")
                continue
        
        return all_datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get information about a specific dataset"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset = self.datasets[dataset_name]
        info = {}
        
        for split, df in dataset.items():
            info[split] = {
                'num_samples': len(df),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_labels': df['label'].nunique() if 'label' in df.columns else None,
                'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else None
            }
        
        return info
    
    def create_combined_dataset(self, dataset_names: List[str], 
                              stratify: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Combine multiple datasets into one
        
        Args:
            dataset_names: List of dataset names to combine
            stratify: Whether to maintain label distribution when combining
            
        Returns:
            Combined dataset with train/validation/test splits
        """
        logger.info(f"Combining datasets: {dataset_names}")
        
        combined_data = {'train': [], 'validation': [], 'test': []}
        
        for dataset_name in dataset_names:
            if dataset_name not in self.datasets:
                self.load_dataset(dataset_name)
            
            dataset = self.datasets[dataset_name]
            
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    df = dataset[split].copy()
                    df['source_dataset'] = dataset_name
                    combined_data[split].append(df)
        
        # Concatenate all splits
        result = {}
        for split in ['train', 'validation', 'test']:
            if combined_data[split]:
                result[split] = pd.concat(combined_data[split], ignore_index=True)
                logger.info(f"Combined {split}: {len(result[split])} samples")
        
        return result
    
    def prepare_for_training(self, dataset_name: str, 
                           text_column: str = 'text',
                           label_column: str = 'label') -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Prepare dataset for training by extracting text and labels
        
        Args:
            dataset_name: Name of the dataset
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Dictionary with train/validation/test splits as (texts, labels) tuples
        """
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset = self.datasets[dataset_name]
        prepared_data = {}
        
        for split, df in dataset.items():
            # Handle missing columns gracefully
            if text_column not in df.columns:
                # Try to find text column with different names
                text_candidates = ['text', 'document', 'content', 'description', 'case_text']
                text_column = next((col for col in text_candidates if col in df.columns), None)
                
                if text_column is None:
                    raise ValueError(f"No text column found in {dataset_name} {split}")
            
            if label_column not in df.columns:
                # Try to find label column with different names
                label_candidates = ['label', 'category', 'class', 'type', 'classification']
                label_column = next((col for col in label_candidates if col in df.columns), None)
                
                if label_column is None:
                    raise ValueError(f"No label column found in {dataset_name} {split}")
            
            # Extract text and labels
            texts = df[text_column].astype(str).tolist()
            labels = df[label_column].astype(str).tolist()
            
            # Remove any NaN values
            valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels)) 
                           if pd.notna(text) and pd.notna(label) and text.strip() and label.strip()]
            
            texts = [texts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            prepared_data[split] = (texts, labels)
            logger.info(f"Prepared {split}: {len(texts)} samples")
        
        return prepared_data
    
    def get_label_mapping(self, dataset_name: str, label_column: str = 'label') -> Dict[str, int]:
        """Get label to integer mapping for a dataset"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        all_labels = set()
        for split, df in self.datasets[dataset_name].items():
            if label_column in df.columns:
                all_labels.update(df[label_column].unique())
        
        # Remove NaN values
        all_labels = {label for label in all_labels if pd.notna(label)}
        
        # Create mapping
        label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        return label_to_id
    
    def validate_dataset_format(self, df: pd.DataFrame, 
                              required_columns: List[str] = None) -> bool:
        """Validate that dataset has required format"""
        if required_columns is None:
            required_columns = ['text', 'label']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for empty values
        for col in required_columns:
            if df[col].isnull().any():
                logger.warning(f"Found null values in column: {col}")
        
        return True
    
    def split_dataset(self, df: pd.DataFrame, 
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     stratify_column: str = 'label',
                     random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Split a dataset into train/validation/test sets"""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size,
            stratify=df[stratify_column] if stratify_column in df.columns else None,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=train_val[stratify_column] if stratify_column in train_val.columns else None,
            random_state=random_state
        )
        
        logger.info(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return {
            'train': train,
            'validation': val,
            'test': test
        }

# Utility functions
def load_single_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a single CSV file with error handling"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV: {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def save_processed_dataset(dataset: Dict[str, pd.DataFrame], 
                         dataset_name: str,
                         output_dir: str = "data/processed") -> None:
    """Save processed dataset splits"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    for split, df in dataset.items():
        file_path = output_path / f"{dataset_name}_{split}_processed.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {split} split to {file_path}")

if __name__ == "__main__":
    # Example usage
    loader = LegalDatasetLoader()
    
    # Load SCOTUS dataset
    scotus_data = loader.load_dataset("scotus")
    
    # Get dataset info
    info = loader.get_dataset_info("scotus")
    print("SCOTUS Dataset Info:", info)
    
    # Prepare for training
    prepared_data = loader.prepare_for_training("scotus")
    print("Prepared data keys:", prepared_data.keys())