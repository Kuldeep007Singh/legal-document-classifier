from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request schema for single document prediction."""
    text: str = Field(..., description="Document text to classify", min_length=1, max_length=50000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v

class PredictionResponse(BaseModel):
    """Response schema for single document prediction."""
    predicted_class: str = Field(..., description="Predicted document class")
    confidence: float = Field(..., description="Confidence score for prediction", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class BatchPredictionRequest(BaseModel):
    """Request schema for batch document prediction."""
    texts: List[str] = Field(..., description="List of document texts to classify", min_items=1, max_items=100)
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError('All texts must be non-empty')
        return v

class BatchPredictionResponse(BaseModel):
    """Response schema for batch document prediction."""
    results: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of documents processed")

class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_type: str = Field(..., description="Type of the model")
    num_classes: int = Field(..., description="Number of classes the model can predict")
    model_name: Optional[str] = Field(None, description="Name of the base model")
    max_length: Optional[int] = Field(None, description="Maximum input length")
    device: str = Field(..., description="Device the model is running on")
    total_parameters: int = Field(..., description="Total number of model parameters")
    trainable_parameters: int = Field(..., description="Number of trainable parameters")
    class_labels: Optional[List[str]] = Field(None, description="List of class labels")

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service health status")
    model_status: str = Field(..., description="Model loading status")
    device: str = Field(..., description="Computing device being used")
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)

class FileUploadResponse(BaseModel):
    """Response schema for file upload prediction."""
    filename: str = Field(..., description="Name of the uploaded file")
    predicted_class: str = Field(..., description="Predicted document class")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    document_length: int = Field(..., description="Length of extracted text")

class TrainingRequest(BaseModel):
    """Request schema for model training (if training endpoint is implemented)."""
    dataset_name: str = Field(..., description="Name of the dataset to train on")
    model_config: Dict[str, Any] = Field({}, description="Model configuration parameters")
    training_config: Dict[str, Any] = Field({}, description="Training configuration parameters")

class TrainingResponse(BaseModel):
    """Response schema for training status."""
    training_id: str = Field(..., description="Unique identifier for the training job")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.now)
