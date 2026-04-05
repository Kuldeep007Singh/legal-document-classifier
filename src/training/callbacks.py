# src/training/callbacks.py

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingCallbacks:
    """Handles training callbacks and monitoring"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.callbacks = []
        
        # Initialize callbacks based on config
        self.early_stopping = config.get('early_stopping', False)
        self.save_best_model = config.get('save_best_model', True)
        self.log_metrics = config.get('log_metrics', True)
        
        # Tracking variables
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 5)
        
        logger.info("TrainingCallbacks initialized")
    
    def on_epoch_start(self, epoch: int, logs: Dict = None):
        """Called at the start of each epoch"""
        logger.info(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Called at the end of each epoch"""
        if logs:
            logger.info(f"Epoch {epoch} completed - Metrics: {logs}")
            
            # Early stopping logic
            if self.early_stopping:
                current_score = logs.get('val_accuracy', logs.get('val_loss', 0))
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    return True  # Signal to stop training
        
        return False
    
    def on_batch_start(self, batch: int, logs: Dict = None):
        """Called at the start of each batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        """Called at the end of each batch"""
        pass
    
    def on_train_start(self, logs: Dict = None):
        """Called at the start of training"""
        logger.info("Training started")
    
    def on_train_end(self, logs: Dict = None):
        """Called at the end of training"""
        logger.info("Training completed")
        if logs:
            logger.info(f"Final metrics: {logs}")
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped"""
        return self.patience_counter >= self.patience if self.early_stopping else False



