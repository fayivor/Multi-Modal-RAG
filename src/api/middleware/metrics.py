"""Metrics collection middleware for Prometheus monitoring."""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.config import settings

# Prometheus metrics
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

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

# RAG-specific metrics
SEARCH_REQUESTS = Counter(
    'rag_search_requests_total',
    'Total search requests',
    ['modality', 'intent']
)

SEARCH_DURATION = Histogram(
    'rag_search_duration_seconds',
    'Search request duration in seconds',
    ['modality', 'reranked']
)

DOCUMENT_UPLOADS = Counter(
    'rag_document_uploads_total',
    'Total document uploads',
    ['modality', 'status']
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    'rag_document_processing_duration_seconds',
    'Document processing duration in seconds',
    ['modality']
)

LLM_REQUESTS = Counter(
    'rag_llm_requests_total',
    'Total LLM requests',
    ['provider', 'model']
)

LLM_DURATION = Histogram(
    'rag_llm_duration_seconds',
    'LLM request duration in seconds',
    ['provider', 'model']
)

LLM_TOKENS = Histogram(
    'rag_llm_tokens_total',
    'Total LLM tokens used',
    ['provider', 'model', 'type']  # type: input/output
)

EMBEDDING_REQUESTS = Counter(
    'rag_embedding_requests_total',
    'Total embedding requests',
    ['modality', 'model']
)

EMBEDDING_DURATION = Histogram(
    'rag_embedding_duration_seconds',
    'Embedding request duration in seconds',
    ['modality', 'model']
)

VECTOR_STORE_OPERATIONS = Counter(
    'rag_vector_store_operations_total',
    'Total vector store operations',
    ['operation', 'status']  # operation: search/upsert/delete, status: success/error
)

VECTOR_STORE_DURATION = Histogram(
    'rag_vector_store_duration_seconds',
    'Vector store operation duration in seconds',
    ['operation']
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP and application metrics."""
    
    def __init__(self, app):
        """Initialize metrics middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.enabled = settings.monitoring.enable_metrics
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for request/response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        if not self.enabled:
            return await call_next(request)
        
        start_time = time.time()
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        try:
            # Get request size
            request_size = self._get_request_size(request)
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get response size
            response_size = self._get_response_size(response)
            
            # Extract endpoint pattern
            endpoint = self._get_endpoint_pattern(request)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            REQUEST_SIZE.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(request_size)
            
            RESPONSE_SIZE.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(response_size)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            endpoint = self._get_endpoint_pattern(request)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
            
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _get_request_size(self, request: Request) -> int:
        """Get request size in bytes.
        
        Args:
            request: HTTP request
            
        Returns:
            Request size in bytes
        """
        content_length = request.headers.get("content-length")
        return int(content_length) if content_length else 0
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes.
        
        Args:
            response: HTTP response
            
        Returns:
            Response size in bytes
        """
        if hasattr(response, 'body') and response.body:
            return len(response.body)
        return 0
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Extract endpoint pattern from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Endpoint pattern string
        """
        # Get the route pattern if available
        if hasattr(request, 'scope') and 'route' in request.scope:
            route = request.scope['route']
            if hasattr(route, 'path'):
                return route.path
        
        # Fallback to path with parameter normalization
        path = request.url.path
        
        # Normalize common patterns
        import re
        
        # Replace UUIDs with placeholder
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Replace numeric IDs with placeholder
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path


class RAGMetricsCollector:
    """Collector for RAG-specific metrics."""
    
    @staticmethod
    def record_search_request(
        modality: str,
        intent: str,
        duration: float,
        reranked: bool = False
    ) -> None:
        """Record search request metrics.
        
        Args:
            modality: Search modality
            intent: Query intent
            duration: Search duration in seconds
            reranked: Whether results were reranked
        """
        if not settings.monitoring.enable_metrics:
            return
        
        SEARCH_REQUESTS.labels(
            modality=modality,
            intent=intent
        ).inc()
        
        SEARCH_DURATION.labels(
            modality=modality,
            reranked=str(reranked).lower()
        ).observe(duration)
    
    @staticmethod
    def record_document_upload(
        modality: str,
        status: str,
        processing_duration: float
    ) -> None:
        """Record document upload metrics.
        
        Args:
            modality: Document modality
            status: Upload status (success/error)
            processing_duration: Processing duration in seconds
        """
        if not settings.monitoring.enable_metrics:
            return
        
        DOCUMENT_UPLOADS.labels(
            modality=modality,
            status=status
        ).inc()
        
        if status == "success":
            DOCUMENT_PROCESSING_DURATION.labels(
                modality=modality
            ).observe(processing_duration)
    
    @staticmethod
    def record_llm_request(
        provider: str,
        model: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """Record LLM request metrics.
        
        Args:
            provider: LLM provider
            model: Model name
            duration: Request duration in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if not settings.monitoring.enable_metrics:
            return
        
        LLM_REQUESTS.labels(
            provider=provider,
            model=model
        ).inc()
        
        LLM_DURATION.labels(
            provider=provider,
            model=model
        ).observe(duration)
        
        if input_tokens > 0:
            LLM_TOKENS.labels(
                provider=provider,
                model=model,
                type="input"
            ).observe(input_tokens)
        
        if output_tokens > 0:
            LLM_TOKENS.labels(
                provider=provider,
                model=model,
                type="output"
            ).observe(output_tokens)
    
    @staticmethod
    def record_embedding_request(
        modality: str,
        model: str,
        duration: float
    ) -> None:
        """Record embedding request metrics.
        
        Args:
            modality: Content modality
            model: Embedding model name
            duration: Request duration in seconds
        """
        if not settings.monitoring.enable_metrics:
            return
        
        EMBEDDING_REQUESTS.labels(
            modality=modality,
            model=model
        ).inc()
        
        EMBEDDING_DURATION.labels(
            modality=modality,
            model=model
        ).observe(duration)
    
    @staticmethod
    def record_vector_store_operation(
        operation: str,
        status: str,
        duration: float
    ) -> None:
        """Record vector store operation metrics.
        
        Args:
            operation: Operation type (search/upsert/delete)
            status: Operation status (success/error)
            duration: Operation duration in seconds
        """
        if not settings.monitoring.enable_metrics:
            return
        
        VECTOR_STORE_OPERATIONS.labels(
            operation=operation,
            status=status
        ).inc()
        
        if status == "success":
            VECTOR_STORE_DURATION.labels(
                operation=operation
            ).observe(duration)


# Global metrics collector instance
rag_metrics = RAGMetricsCollector()
