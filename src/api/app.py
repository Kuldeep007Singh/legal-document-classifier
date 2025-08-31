from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import torch
from pathlib import Path
import tempfile
import logging
from typing import List, Dict, Optional, Any
import asyncio
from contextlib import asynccontextmanager

from .routes import router
from .schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest
from .middleware import setup_middleware
from ..models.transformer_models import TransformerClassifier
from ..data.document_parser import DocumentParser
from ..inference.predictor import DocumentClassifier
from ..utils.logging_utils import setup_logger
from ..config.settings import settings

# Setup logging
logger = setup_logger(__name__)

# Global variables for model loading
model_instance = None
document_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up Legal Document Classifier API")
    
    # Load model
    global model_instance, document_classifier
    try:
        model_path = settings.MODEL_DIR / "best_models" / "best_model.pt"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            
            # Load the model
            model_instance = TransformerClassifier.load_model(str(model_path))
            model_instance.eval()
            
            # Initialize document classifier
            document_classifier = DocumentClassifier(
                model=model_instance,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}. Some endpoints will not work.")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Document Classifier API")

# Create FastAPI app
app = FastAPI(
    title="Legal Document Classifier",
    description="AI-powered legal document classification system using LexGLUE datasets",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(router, prefix="/api/v1")

# Security
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token validation (implement proper authentication as needed)."""
    # For demo purposes - implement proper JWT validation in production
    if credentials.credentials == settings.SECRET_KEY:
        return {"user_id": "demo_user"}
    raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal Document Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model_instance is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_document(request: PredictionRequest, user=Depends(get_current_user)):
    """Classify a single document."""
    if document_classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        result = document_classifier.predict_text(request.text)
        
        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=result.get('processing_time', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """Classify an uploaded document file."""
    if document_classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
        )
    
    # Check file size
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    if file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file.size} exceeds maximum allowed size of {max_size} bytes"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Parse and classify document
        result = document_classifier.predict_file(tmp_file_path)
        
        # Clean up temporary file
        Path(tmp_file_path).unlink()
        
        return {
            "filename": file.filename,
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "processing_time": result.get('processing_time', 0.0),
            "document_length": len(result.get('text', ''))
        }
        
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(
    request: BatchPredictionRequest,
    user=Depends(get_current_user)
):
    """Classify multiple documents in batch."""
    if document_classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 documents")
    
    try:
        results = []
        
        for i, text in enumerate(request.texts):
            result = document_classifier.predict_text(text)
            
            results.append({
                "index": i,
                "predicted_class": result['predicted_class'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities']
            })
        
        return {
            "results": results,
            "total_processed": len(results)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info(user=Depends(get_current_user)):
    """Get information about the loaded model."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model information
        info = {
            "model_type": model_instance.__class__.__name__,
            "num_classes": model_instance.num_classes,
            "model_name": getattr(model_instance, 'model_name', 'unknown'),
            "max_length": getattr(model_instance, 'max_length', 512),
            "device": str(next(model_instance.parameters()).device),
            "total_parameters": sum(p.numel() for p in model_instance.parameters()),
            "trainable_parameters": sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        }
        
        # Add class labels if available
        if hasattr(document_classifier, 'label_names'):
            info["class_labels"] = document_classifier.label_names
        
        return info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
