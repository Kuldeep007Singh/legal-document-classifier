# Code Path: src/models/transformer_models.py

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertForSequenceClassification, RobertaForSequenceClassification,
    DistilBertForSequenceClassification, TrainingArguments, Trainer
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    """Custom model output container"""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None

class LegalBERTClassifier(nn.Module):
    """Legal document classifier based on BERT"""
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 hidden_size: Optional[int] = None):
        
        super(LegalBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        classifier_hidden_size = hidden_size or self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_size, num_classes)
        )
        
        self.loss_fn = CrossEntropyLoss()
        
        logger.info(f"Initialized LegalBERTClassifier with {model_name}, {num_classes} classes")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> ModelOutput:
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        # Get predictions and probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        if return_dict:
            return ModelOutput(
                logits=logits,
                loss=loss,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                predictions=predictions,
                probabilities=probabilities
            )
        
        return logits, loss

class HierarchicalLegalClassifier(nn.Module):
    """Hierarchical classifier for legal documents with section-aware attention"""
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 max_sections: int = 10,
                 section_hidden_size: int = 256,
                 dropout_rate: float = 0.1):
        
        super(HierarchicalLegalClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_sections = max_sections
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Section-level processing
        self.section_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Document-level processing
        self.document_lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=section_hidden_size,
            num_layers=2,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        
        # Final classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(section_hidden_size * 2, num_classes)
        
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                section_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> ModelOutput:
        
        batch_size, seq_len = input_ids.shape
        
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Section-aware processing
        if section_mask is not None:
            # Apply section-level attention
            attended_output, _ = self.section_attention(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=~attention_mask.bool()
            )
        else:
            attended_output = sequence_output
        
        # Global average pooling with attention mask
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(attended_output.size()).float()
        sum_embeddings = torch.sum(attended_output * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-7)
        pooled_output = sum_embeddings / sum_mask
        
        # Document-level LSTM
        lstm_output, _ = self.document_lstm(pooled_output.unsqueeze(1))
        document_repr = lstm_output.squeeze(1)  # Remove sequence dimension
        
        # Final classification
        document_repr = self.dropout(document_repr)
        logits = self.classifier(document_repr)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        # Get predictions and probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            predictions=predictions,
            probabilities=probabilities
        )

class LegalRoBERTaClassifier(nn.Module):
    """RoBERTa-based legal document classifier"""
    
    def __init__(self,
                 model_name: str = "roberta-base",
                 num_classes: int = 2,
                 dropout_rate: float = 0.1):
        
        super(LegalRoBERTaClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Use pre-built RoBERTa for sequence classification
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        logger.info(f"Initialized LegalRoBERTaClassifier with {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        return ModelOutput(
            logits=outputs.logits,
            loss=outputs.loss,
            predictions=predictions,
            probabilities=probabilities
        )

class EnsembleTransformerClassifier(nn.Module):
    """Ensemble of multiple transformer models"""
    
    def __init__(self,
                 model_configs: List[Dict[str, Any]],
                 num_classes: int = 2,
                 ensemble_method: str = "average"):
        
        super(EnsembleTransformerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        self.models = nn.ModuleList()
        
        # Initialize individual models
        for config in model_configs:
            model_type = config.get('type', 'bert')
            model_name = config.get('model_name', 'bert-base-uncased')
            
            if model_type == 'bert':
                model = LegalBERTClassifier(
                    model_name=model_name,
                    num_classes=num_classes,
                    dropout_rate=config.get('dropout_rate', 0.1)
                )
            elif model_type == 'roberta':
                model = LegalRoBERTaClassifier(
                    model_name=model_name,
                    num_classes=num_classes,
                    dropout_rate=config.get('dropout_rate', 0.1)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.models.append(model)
        
        # Ensemble weights
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(len(model_configs)) / len(model_configs))
        
        logger.info(f"Initialized ensemble with {len(self.models)} models")
    
    def forward(self, input_ids, attention_mask, labels=None):
        model_outputs = []
        total_loss = 0.0
        
        # Get outputs from all models
        for model in self.models:
            output = model(input_ids, attention_mask, labels)
            model_outputs.append(output)
            
            if output.loss is not None:
                total_loss += output.loss
        
        # Combine predictions
        if self.ensemble_method == "average":
            ensemble_logits = torch.stack([out.logits for out in model_outputs]).mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = torch.softmax(self.ensemble_weights, dim=0)
            ensemble_logits = sum(w * out.logits for w, out in zip(weights, model_outputs))
        else:
            # Voting (for discrete predictions)
            predictions = torch.stack([out.predictions for out in model_outputs])
            ensemble_logits = torch.mode(predictions, dim=0)[0]
        
        # Calculate ensemble loss
        ensemble_loss = None
        if labels is not None:
            ensemble_loss = total_loss / len(self.models)
        
        probabilities = torch.softmax(ensemble_logits, dim=-1)
        predictions = torch.argmax(ensemble_logits, dim=-1)
        
        return ModelOutput(
            logits=ensemble_logits,
            loss=ensemble_loss,
            predictions=predictions,
            probabilities=probabilities
        )

class SpecializedLegalBERT(nn.Module):
    """Specialized BERT model for legal documents with domain-specific modifications"""
    
    def __init__(self,
                 model_name: str = "nlpaueb/legal-bert-base-uncased",
                 num_classes: int = 2,
                 use_legal_attention: bool = True,
                 dropout_rate: float = 0.1):
        
        super(SpecializedLegalBERT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_legal_attention = use_legal_attention
        
        # Load Legal-BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Legal-specific attention layer
        if use_legal_attention:
            self.legal_attention = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Multi-layer classifier with legal domain adaptation
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.loss_fn = CrossEntropyLoss()
        
        logger.info(f"Initialized SpecializedLegalBERT with {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply legal-specific attention if enabled
        if self.use_legal_attention:
            attended_output, attention_weights = self.legal_attention(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=~attention_mask.bool()
            )
            
            # Combine original and attended representations
            combined_output = sequence_output + attended_output
        else:
            combined_output = sequence_output
        
        # Global max pooling over sequence dimension
        pooled_output = torch.max(combined_output, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions=predictions,
            probabilities=probabilities
        )

class MultiTaskLegalClassifier(nn.Module):
    """Multi-task classifier for different legal document types"""
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 task_configs: Dict[str, int] = None,
                 shared_hidden_size: int = 512,
                 dropout_rate: float = 0.1):
        
        super(MultiTaskLegalClassifier, self).__init__()
        
        self.model_name = model_name
        self.task_configs = task_configs or {"main_task": 2}
        
        # Shared BERT encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, shared_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in self.task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(shared_hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.loss_fn = CrossEntropyLoss()
        
        logger.info(f"Initialized MultiTaskLegalClassifier with tasks: {list(self.task_configs.keys())}")
    
    def forward(self, input_ids, attention_mask, labels=None, task_name="main_task"):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Shared representation
        shared_repr = self.shared_layer(pooled_output)
        
        # Task-specific prediction
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")
        
        logits = self.task_heads[task_name](shared_repr)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            predictions=predictions,
            probabilities=probabilities
        )

class ModelFactory:
    """Factory class to create different types of models"""
    
    @staticmethod
    def create_model(model_type: str,
                    model_name: str,
                    num_classes: int,
                    **kwargs) -> nn.Module:
        """Create a model based on type and configuration"""
        
        if model_type == "legal_bert":
            return LegalBERTClassifier(
                model_name=model_name,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == "hierarchical":
            return HierarchicalLegalClassifier(
                model_name=model_name,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == "roberta":
            return LegalRoBERTaClassifier(
                model_name=model_name,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == "multitask":
            return MultiTaskLegalClassifier(
                model_name=model_name,
                task_configs=kwargs.get('task_configs'),
                **kwargs
            )
        elif model_type == "ensemble":
            return EnsembleTransformerClassifier(
                model_configs=kwargs.get('model_configs'),
                num_classes=num_classes,
                **kwargs
            )
        else:
            # Default to standard BERT
            return BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes
            )

class ModelManager:
    """Manage loading, saving, and switching between different models"""
    
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_models = {}
    
    def save_model(self, 
                   model: nn.Module, 
                   model_name: str,
                   tokenizer: Optional[Any] = None,
                   metadata: Optional[Dict] = None):
        """Save model and associated components"""
        
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True, parents=True)
        
        # Save model state dict
        torch.save(model.state_dict(), model_path / "model.pt")
        
        # Save model config
        if hasattr(model, 'config'):
            model.config.save_pretrained(model_path)
        
        # Save tokenizer
        if tokenizer:
            tokenizer.save_pretrained(model_path)
        
        # Save metadata
        if metadata:
            import json
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, 
                   model_name: str,
                   model_class: type,
                   **model_kwargs) -> Tuple[nn.Module, Any]:
        """Load saved model"""
        
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path / "model.pt"))
        
        # Load tokenizer
        tokenizer = None
        if (model_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.loaded_models[model_name] = (model, tokenizer)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model, tokenizer
    
    def list_saved_models(self) -> List[str]:
        """List all saved models"""
        model_dirs = [d.name for d in self.model_dir.iterdir() 
                     if d.is_dir() and (d / "model.pt").exists()]
        return model_dirs

# Training utilities
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class LegalTrainer(Trainer):
    """Custom trainer for legal document classification"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with class weights if provided"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        if self.class_weights is not None:
            loss_fn = CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            if loss is None:
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(outputs.logits, labels)
        
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Example usage
    
    # Create a sample model
    model = LegalBERTClassifier(
        model_name="bert-base-uncased",
        num_classes=5,
        dropout_rate=0.1
    )
    
    # Test with dummy data
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 5, (batch_size,))
    
    # Forward pass
    output = model(input_ids, attention_mask, labels)
    
    print("Output logits shape:", output.logits.shape)
    print("Loss:", output.loss.item() if output.loss else None)
    print("Predictions:", output.predictions)
    print("Probabilities shape:", output.probabilities.shape)