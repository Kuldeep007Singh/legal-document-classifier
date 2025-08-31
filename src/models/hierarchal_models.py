# Code Path: src/models/hierarchical_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class HierarchicalDocumentClassifier(BaseModel):
    """
    Hierarchical model that processes documents section by section,
    then aggregates section representations for final classification.
    """
    
    def __init__(self, 
                 model_name: str = 'nlpaueb/legal-bert-base-uncased',
                 num_classes: int = 6,
                 max_sections: int = 20,
                 max_section_length: int = 512,
                 hidden_dim: int = 768,
                 dropout: float = 0.1):
        """
        Initialize hierarchical classifier.
        
        Args:
            model_name: Pre-trained model name
            num_classes: Number of document classes
            max_sections: Maximum number of sections per document
            max_section_length: Maximum tokens per section
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_sections = max_sections
        self.max_section_length = max_section_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the hierarchical model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
    def _build_model(self) -> nn.Module:
        """Build the hierarchical neural network model."""
        
        class HierarchicalNet(nn.Module):
            def __init__(self, model_name, num_classes, max_sections, hidden_dim, dropout):
                super().__init__()
                
                # Section encoder (BERT-based)
                self.section_encoder = AutoModel.from_pretrained(model_name)
                self.section_encoder_dim = self.section_encoder.config.hidden_size
                
                # Section-level processing
                self.section_projection = nn.Linear(self.section_encoder_dim, hidden_dim)
                self.section_dropout = nn.Dropout(dropout)
                
                # Document-level processing (attention-based aggregation)
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True
                )
                
                # Document representation
                self.doc_processor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Classification head
                self.classifier = nn.Linear(hidden_dim // 2, num_classes)
                
                # Positional encoding for sections
                self.positional_encoding = nn.Parameter(
                    torch.randn(max_sections, hidden_dim) * 0.1
                )
                
            def forward(self, input_ids, attention_mask, section_mask=None):
                """
                Forward pass through hierarchical model.
                
                Args:
                    input_ids: [batch_size, max_sections, max_section_length]
                    attention_mask: [batch_size, max_sections, max_section_length]
                    section_mask: [batch_size, max_sections] - mask for valid sections
                """
                batch_size, max_sections, max_seq_len = input_ids.shape
                
                # Reshape for section processing
                input_ids_flat = input_ids.view(-1, max_seq_len)
                attention_mask_flat = attention_mask.view(-1, max_seq_len)
                
                # Encode sections
                section_outputs = self.section_encoder(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat
                )
                
                # Get section representations (CLS token)
                section_reprs = section_outputs.last_hidden_state[:, 0, :]  # [batch_size * max_sections, hidden_dim]
                section_reprs = section_reprs.view(batch_size, max_sections, -1)
                
                # Project and add positional encoding
                section_reprs = self.section_projection(section_reprs)
                section_reprs = self.section_dropout(section_reprs)
                section_reprs += self.positional_encoding[:max_sections].unsqueeze(0)
                
                # Apply section mask if provided
                if section_mask is not None:
                    section_reprs = section_reprs * section_mask.unsqueeze(-1)
                
                # Document-level attention
                doc_repr, attention_weights = self.attention(
                    section_reprs, section_reprs, section_reprs,
                    key_padding_mask=~section_mask.bool() if section_mask is not None else None
                )
                
                # Aggregate document representation (mean pooling over sections)
                if section_mask is not None:
                    doc_repr = (doc_repr * section_mask.unsqueeze(-1)).sum(dim=1) / section_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    doc_repr = doc_repr.mean(dim=1)
                
                # Process document representation
                doc_repr = self.doc_processor(doc_repr)
                
                # Classification
                logits = self.classifier(doc_repr)
                
                return {
                    'logits': logits,
                    'attention_weights': attention_weights,
                    'section_representations': section_reprs,
                    'document_representation': doc_repr
                }
        
        return HierarchicalNet(
            self.model_name, self.num_classes, self.max_sections, 
            self.hidden_dim, self.dropout
        )
    
    def prepare_hierarchical_input(self, sections: List[List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare hierarchical input from document sections.
        
        Args:
            sections: List of documents, each containing list of section texts
            
        Returns:
            Dictionary with tokenized inputs
        """
        batch_size = len(sections)
        
        # Initialize tensors
        input_ids = torch.zeros((batch_size, self.max_sections, self.max_section_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_sections, self.max_section_length), dtype=torch.long)
        section_mask = torch.zeros((batch_size, self.max_sections), dtype=torch.float)
        
        for doc_idx, doc_sections in enumerate(sections):
            for sec_idx, section_text in enumerate(doc_sections[:self.max_sections]):
                # Tokenize section
                encoded = self.tokenizer(
                    section_text,
                    max_length=self.max_section_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids[doc_idx, sec_idx, :] = encoded['input_ids'].squeeze()
                attention_mask[doc_idx, sec_idx, :] = encoded['attention_mask'].squeeze()
                section_mask[doc_idx, sec_idx] = 1.0
        
        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'section_mask': section_mask.to(self.device)
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = 10, 
              learning_rate: float = 2e-5) -> Dict[str, Any]:
        """
        Train the hierarchical model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
        )
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.model.train()
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            self.scheduler.step()
        
        return history
    
    def _train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch['inputs'])
            loss = self.criterion(outputs['logits'], batch['labels'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(**batch['inputs'])
                loss = self.criterion(outputs['logits'], batch['labels'])
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        self.model.train()
        return total_loss / len(val_loader), correct / total
    
    def predict(self, sections: List[List[str]]) -> np.ndarray:
        """Make predictions on hierarchical input."""
        self.model.eval()
        
        inputs = self.prepare_hierarchical_input(sections)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs['logits'], dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, sections: List[List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        
        inputs = self.prepare_hierarchical_input(sections)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs['logits'], dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_attention_weights(self, sections: List[List[str]]) -> np.ndarray:
        """Get attention weights for interpretability."""
        self.model.eval()
        
        inputs = self.prepare_hierarchical_input(sections)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_weights = outputs['attention_weights']
        
        return attention_weights.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save the hierarchical model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'max_sections': self.max_sections,
                'max_section_length': self.max_section_length,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout
            }
        }, filepath)
        logger.info(f"Hierarchical model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved hierarchical model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update config
        config = checkpoint['model_config']
        self.__dict__.update(config)
        
        # Rebuild and load model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        logger.info(f"Hierarchical model loaded from {filepath}")