from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI app."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware (configure for production)
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Custom middleware
    app.middleware("http")(log_requests)
    app.middleware("http")(add_process_time_header)

async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
