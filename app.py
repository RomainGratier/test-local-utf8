"""
FastAPI application for computer vision image classification.

This is the main application entry point that sets up the FastAPI server
with all routes, middleware, and configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger
from src.api.routes import router

# Configure logging
setup_logging()
logger = get_logger(__name__)

# Get configuration
config = get_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Computer Vision Image Classification API")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Version: {config.version}")
    
    # Load model if path is provided
    if config.model_path and os.path.exists(config.model_path):
        try:
            from src.api.routes import load_model
            load_model(config.model_path)
            logger.info(f"Model loaded from {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Computer Vision Image Classification API")

# Create FastAPI application
app = FastAPI(
    title="Computer Vision Image Classification API",
    description="A production-ready image classification system with comprehensive error handling and validation",
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your deployment
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Computer Vision Image Classification API",
        "version": config.version,
        "environment": config.environment,
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Health check endpoint (additional to the one in routes)
@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "version": config.version}

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.logging.level.lower(),
        access_log=True
    )
