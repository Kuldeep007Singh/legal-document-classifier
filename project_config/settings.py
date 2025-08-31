# Code Path: config/settings.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data configuration settings"""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    interim_data_path: str = "data/interim"
    external_data_path: str = "data/external"
    
    # Dataset configurations
    datasets: Dict[str, Dict[str, any]] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {
                "scotus": {"train": "scotus_train.csv", "test": "scotus_test.csv", "validation": "scotus_validation.csv"},
                "ledgar": {"train": "ledgar_train.csv", "test": "ledgar_test.csv", "validation": "ledgar_validation.csv"},
                "eurlex": {"train": "eurlex_train.csv", "test": "eurlex_test.csv", "validation": "eurlex_validation.csv"},
                "ecthr_a": {"train": "ecthr_a_train.csv", "test": "ecthr_a_test.csv", "validation": "ecthr_a_validation.csv"},
                "ecthr_b": {"train": "ecthr_b_train.csv", "test": "ecthr_b_test.csv", "validation": "ecthr_b_validation.csv"},
                "case_hold": {"train": "case_hold_train.csv", "test": "case_hold_test.csv", "validation": "case_hold_validation.csv"},
                "unfair_tos": {"train": "unfair_tos_train.csv", "test": "unfair_tos_test.csv", "validation": "unfair_tos_validation.csv"}
            }

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "bert-base-uncased"
    legal_bert_model: str = "nlpaueb/legal-bert-base-uncased"
    max_sequence_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    
    # Model paths
    model_save_path: str = "models/best_models"
    checkpoint_path: str = "models/checkpoints"
    ensemble_path: str = "models/ensemble"

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    n_trials: int = 50

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pdf', '.docx', '.txt', '.doc']

class Settings:
    """Main settings class that loads all configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.api = APIConfig()
        
        # Load configurations from YAML files if they exist
        self._load_yaml_configs()
        
        # Environment variables override
        self._load_env_overrides()
    
    def _load_yaml_configs(self):
        """Load configurations from YAML files"""
        yaml_files = {
            'model_config.yaml': 'model',
            'data_config.yaml': 'data',
            'training_config.yaml': 'training'
        }
        
        for yaml_file, attr_name in yaml_files.items():
            yaml_path = self.config_dir / yaml_file
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Update the configuration object
                    config_obj = getattr(self, attr_name)
                    for key, value in config_data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
    
    def _load_env_overrides(self):
        """Load environment variable overrides"""
        # Model config overrides
        if os.getenv('MODEL_NAME'):
            self.model.model_name = os.getenv('MODEL_NAME')
        if os.getenv('BATCH_SIZE'):
            self.model.batch_size = int(os.getenv('BATCH_SIZE'))
        if os.getenv('LEARNING_RATE'):
            self.model.learning_rate = float(os.getenv('LEARNING_RATE'))
        
        # API config overrides
        if os.getenv('API_HOST'):
            self.api.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.port = int(os.getenv('API_PORT'))
    
    def get_dataset_path(self, dataset_name: str, split: str) -> Path:
        """Get the path for a specific dataset split"""
        if dataset_name not in self.data.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        filename = self.data.datasets[dataset_name].get(split)
        if not filename:
            raise ValueError(f"Split {split} not found for dataset {dataset_name}")
        
        return Path(self.data.raw_data_path) / filename

# Global settings instance
settings = Settings()