from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .schemas import (
    ModelInfoResponse, HealthResponse, TrainingRequest, 
    TrainingResponse, ErrorResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_api_stats():
    """Get API usage statistics."""
    # This would typically connect to a database or metrics store
    return {
        "total_predictions": 0,  # Implement proper tracking
        "uptime": "0 days",
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat()
    }

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported document formats."""
    return {
        "formats": [
            {
                "extension": ".pdf",
                "mime_type": "application/pdf",
                "description": "Portable Document Format"
            },
            {
                "extension": ".docx", 
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "description": "Microsoft Word Document"
            },
            {
                "extension": ".txt",
                "mime_type": "text/plain", 
                "description": "Plain Text File"
            }
        ]
    }

@router.get("/document-types")
async def get_document_types():
    """Get list of supported legal document types."""
    return {
        "document_types": [
            "Non-Disclosure Agreement (NDA)",
            "License Agreement", 
            "Employment Contract",
            "Service Agreement",
            "Terms of Service",
            "Privacy Policy",
            "Partnership Agreement",
            "Lease Agreement",
            "Purchase Agreement",
            "Confidentiality Agreement"
        ]
    }
