"""FastAPI application for the multi-modal RAG system.

Note: This implementation demonstrates production-grade architectural patterns
and engineering methodologies while maintaining confidentiality of proprietary
business logic and client-specific implementations.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..core.config import settings
from ..core.exceptions import MultiModalRAGException, create_error_response
from .middleware.auth import AuthMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.metrics import MetricsMiddleware
from .routes import documents, health, search

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting multi-modal RAG application")
    
    try:
        # Initialize components
        from ..embeddings.multi_modal import multi_modal_system
        from ..retrieval.vector_store import vector_store
        
        # Preload embedding models if configured
        if settings.embeddings.device != "cpu":
            logger.info("Preloading embedding models")
            await multi_modal_system.preload_all_encoders()
        
        # Check vector store health
        if await vector_store.health_check():
            logger.info("Vector store connection established")
        else:
            logger.warning("Vector store health check failed")
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down multi-modal RAG application")
    
    try:
        # Cleanup resources
        from ..embeddings.multi_modal import multi_modal_system
        multi_modal_system.embedding_manager.unload_all_encoders()
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url=settings.docs_url if settings.enable_docs else None,
    redoc_url=settings.redoc_url if settings.enable_docs else None,
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)

# Add authentication middleware if enabled
if hasattr(settings, 'enable_auth') and settings.enable_auth:
    app.add_middleware(AuthMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Add process time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(MultiModalRAGException)
async def rag_exception_handler(request: Request, exc: MultiModalRAGException):
    """Handle custom RAG exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "RAG exception occurred",
        error_code=exc.error_code,
        error_message=exc.message,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    
    error_response = create_error_response(exc, request_id)
    
    # Determine status code based on exception type
    status_code = 500
    if "validation" in exc.error_code.lower():
        status_code = 422
    elif "authentication" in exc.error_code.lower():
        status_code = 401
    elif "authorization" in exc.error_code.lower():
        status_code = 403
    elif "not_found" in exc.error_code.lower():
        status_code = 404
    elif "rate_limit" in exc.error_code.lower():
        status_code = 429
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal server error occurred",
            "request_id": request_id,
        }
    )


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "status": "running",
        "docs_url": settings.docs_url if settings.enable_docs else None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.monitoring.enable_metrics:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics not enabled"}
        )
    
    from prometheus_client import CONTENT_TYPE_LATEST
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/info")
async def info():
    """Application information endpoint."""
    from ..embeddings.multi_modal import multi_modal_system
    from ..retrieval.vector_store import vector_store
    
    try:
        # Get system information
        embedding_info = multi_modal_system.get_system_info()
        vector_store_info = await vector_store.get_collection_info()
        
        return {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "debug": settings.debug,
            },
            "embeddings": embedding_info,
            "vector_store": {
                "type": settings.vector_store.type.value,
                "collection_info": vector_store_info,
            },
            "llm": {
                "provider": settings.llm.provider.value,
                "model": getattr(settings.llm, f"{settings.llm.provider.value}_model"),
            },
            "features": {
                "cors_enabled": settings.enable_cors,
                "docs_enabled": settings.enable_docs,
                "metrics_enabled": settings.monitoring.enable_metrics,
                "caching_enabled": settings.cache.enable_cache,
            }
        }
        
    except Exception as e:
        logger.error("Failed to get application info", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve application information"}
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
    )
