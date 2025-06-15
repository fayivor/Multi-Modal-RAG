"""Health check API routes."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict

import structlog
from fastapi import APIRouter, HTTPException, Request

from ...core.config import settings
from ...embeddings.multi_modal import multi_modal_system
from ...generation.llm_client import llm_client
from ...retrieval.vector_store import vector_store
from ..models.responses import HealthCheckResponse, SystemInfoResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check(request: Request) -> HealthCheckResponse:
    """Basic health check endpoint."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Basic health check
        status = "healthy"
        timestamp = datetime.utcnow().isoformat()
        
        # Check core components
        components = {}
        
        # Check vector store
        try:
            vector_store_healthy = await vector_store.health_check()
            components["vector_store"] = {
                "status": "healthy" if vector_store_healthy else "unhealthy",
                "type": settings.vector_store.type.value,
                "host": settings.vector_store.qdrant_host,
                "port": settings.vector_store.qdrant_port,
            }
        except Exception as e:
            components["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        # Check embedding system
        try:
            embedding_info = multi_modal_system.get_system_info()
            components["embeddings"] = {
                "status": "healthy",
                "loaded_modalities": embedding_info["loaded_modalities"],
                "registered_modalities": embedding_info["registered_modalities"],
                "is_fitted": embedding_info["is_fitted"],
            }
        except Exception as e:
            components["embeddings"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        # Check LLM client
        try:
            llm_info = llm_client.get_model_info()
            components["llm"] = {
                "status": "healthy",
                "provider": llm_info["provider"],
                "model_name": llm_info["model_name"],
            }
        except Exception as e:
            components["llm"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        response = HealthCheckResponse(
            status=status,
            timestamp=timestamp,
            version=settings.app_version,
            components=components,
        )
        
        logger.info(
            "Health check completed",
            status=status,
            components_checked=len(components),
            request_id=request_id,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Health check failed",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(request: Request) -> HealthCheckResponse:
    """Detailed health check with performance metrics."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    start_time = time.time()
    
    try:
        status = "healthy"
        timestamp = datetime.utcnow().isoformat()
        components = {}
        
        # Detailed vector store check
        try:
            vector_store_start = time.time()
            vector_store_healthy = await vector_store.health_check()
            vector_store_time = (time.time() - vector_store_start) * 1000
            
            if vector_store_healthy:
                collection_info = await vector_store.get_collection_info()
                components["vector_store"] = {
                    "status": "healthy",
                    "response_time_ms": vector_store_time,
                    "collection_info": collection_info,
                }
            else:
                components["vector_store"] = {
                    "status": "unhealthy",
                    "response_time_ms": vector_store_time,
                }
                status = "degraded"
                
        except Exception as e:
            components["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        # Detailed embedding system check
        try:
            embedding_start = time.time()
            embedding_info = multi_modal_system.get_system_info()
            embedding_time = (time.time() - embedding_start) * 1000
            
            components["embeddings"] = {
                "status": "healthy",
                "response_time_ms": embedding_time,
                "system_info": embedding_info,
            }
        except Exception as e:
            components["embeddings"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        # Test LLM with a simple query
        try:
            llm_start = time.time()
            test_response = await llm_client.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
            )
            llm_time = (time.time() - llm_start) * 1000
            
            components["llm"] = {
                "status": "healthy",
                "response_time_ms": llm_time,
                "test_response": test_response[:50] + "..." if len(test_response) > 50 else test_response,
                "model_info": llm_client.get_model_info(),
            }
        except Exception as e:
            components["llm"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status = "degraded"
        
        # Check Redis cache if enabled
        if settings.cache.enable_cache:
            try:
                # This would check Redis connection
                # For now, just mark as healthy
                components["cache"] = {
                    "status": "healthy",
                    "enabled": True,
                    "url": settings.redis.url,
                }
            except Exception as e:
                components["cache"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                status = "degraded"
        
        # Performance metrics
        total_time = (time.time() - start_time) * 1000
        metrics = {
            "total_check_time_ms": total_time,
            "components_checked": len(components),
            "healthy_components": sum(1 for c in components.values() if c["status"] == "healthy"),
        }
        
        response = HealthCheckResponse(
            status=status,
            timestamp=timestamp,
            version=settings.app_version,
            components=components,
            metrics=metrics,
        )
        
        logger.info(
            "Detailed health check completed",
            status=status,
            total_time_ms=total_time,
            request_id=request_id,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Detailed health check failed",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Detailed health check failed")


@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, str]:
    """Kubernetes readiness probe endpoint."""
    try:
        # Check if critical components are ready
        vector_store_ready = await vector_store.health_check()
        
        if not vector_store_ready:
            raise HTTPException(status_code=503, detail="Vector store not ready")
        
        return {"status": "ready"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@router.get("/info", response_model=SystemInfoResponse)
async def system_info(request: Request) -> SystemInfoResponse:
    """Get detailed system information."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Application info
        application = {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": settings.app_description,
            "environment": settings.environment,
            "debug": settings.debug,
            "log_level": settings.log_level,
        }
        
        # Embedding system info
        embeddings = multi_modal_system.get_system_info()
        
        # Vector store info
        vector_store_info = {
            "type": settings.vector_store.type.value,
            "host": settings.vector_store.qdrant_host,
            "port": settings.vector_store.qdrant_port,
        }
        
        try:
            collection_info = await vector_store.get_collection_info()
            vector_store_info["collection"] = collection_info
        except Exception as e:
            vector_store_info["collection_error"] = str(e)
        
        # LLM info
        llm = {
            "provider": settings.llm.provider.value,
            "model": getattr(settings.llm, f"{settings.llm.provider.value}_model"),
            "max_tokens": getattr(settings.llm, f"{settings.llm.provider.value}_max_tokens", None),
        }
        
        # Features
        features = {
            "cors_enabled": settings.enable_cors,
            "docs_enabled": settings.enable_docs,
            "metrics_enabled": settings.monitoring.enable_metrics,
            "caching_enabled": settings.cache.enable_cache,
        }
        
        # Performance stats (if available)
        performance = {
            "memory_usage": embeddings.get("memory_usage", {}),
        }
        
        response = SystemInfoResponse(
            application=application,
            embeddings=embeddings,
            vector_store=vector_store_info,
            llm=llm,
            features=features,
            performance=performance,
        )
        
        logger.info(
            "System info retrieved",
            request_id=request_id,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Failed to get system info",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve system information")
