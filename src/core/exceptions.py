"""Custom exceptions for the multi-modal RAG system."""

from typing import Any, Dict, Optional


class MultiModalRAGException(Exception):
    """Base exception for all multi-modal RAG errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(MultiModalRAGException):
    """Raised when there's a configuration error."""
    pass


class ValidationError(MultiModalRAGException):
    """Raised when input validation fails."""
    pass


# Ingestion errors
class IngestionError(MultiModalRAGException):
    """Base class for ingestion-related errors."""
    pass


class FileNotFoundError(IngestionError):
    """Raised when a file cannot be found."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when trying to process an unsupported file type."""
    pass


class FileSizeExceededError(IngestionError):
    """Raised when file size exceeds the maximum allowed size."""
    pass


class DocumentParsingError(IngestionError):
    """Raised when document parsing fails."""
    pass


class ImageProcessingError(IngestionError):
    """Raised when image processing fails."""
    pass


class CodeAnalysisError(IngestionError):
    """Raised when code analysis fails."""
    pass


# Embedding errors
class EmbeddingError(MultiModalRAGException):
    """Base class for embedding-related errors."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when a model fails to load."""
    pass


class EncodingError(EmbeddingError):
    """Raised when encoding fails."""
    pass


class ModelNotFoundError(EmbeddingError):
    """Raised when a requested model is not found."""
    pass


# Retrieval errors
class RetrievalError(MultiModalRAGException):
    """Base class for retrieval-related errors."""
    pass


class VectorStoreError(RetrievalError):
    """Raised when vector store operations fail."""
    pass


class SearchError(RetrievalError):
    """Raised when search operations fail."""
    pass


class RerankingError(RetrievalError):
    """Raised when reranking fails."""
    pass


class QueryProcessingError(RetrievalError):
    """Raised when query processing fails."""
    pass


# Generation errors
class GenerationError(MultiModalRAGException):
    """Base class for generation-related errors."""
    pass


class LLMError(GenerationError):
    """Raised when LLM operations fail."""
    pass


class PromptTooLongError(GenerationError):
    """Raised when prompt exceeds token limits."""
    pass


class ResponseGenerationError(GenerationError):
    """Raised when response generation fails."""
    pass


class CitationError(GenerationError):
    """Raised when citation tracking fails."""
    pass


# API errors
class APIError(MultiModalRAGException):
    """Base class for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


class RateLimitExceededError(APIError):
    """Raised when rate limits are exceeded."""
    pass


class RequestValidationError(APIError):
    """Raised when request validation fails."""
    pass


# Cache errors
class CacheError(MultiModalRAGException):
    """Base class for cache-related errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass


class CacheOperationError(CacheError):
    """Raised when cache operations fail."""
    pass


# Database errors
class DatabaseError(MultiModalRAGException):
    """Base class for database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when database queries fail."""
    pass


# Monitoring errors
class MonitoringError(MultiModalRAGException):
    """Base class for monitoring-related errors."""
    pass


class MetricsError(MonitoringError):
    """Raised when metrics collection fails."""
    pass


class HealthCheckError(MonitoringError):
    """Raised when health checks fail."""
    pass


# Utility functions for error handling
def create_error_response(
    exception: MultiModalRAGException,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized error response.
    
    Args:
        exception: The exception to convert
        request_id: Optional request ID for tracking
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "error": exception.error_code,
        "message": exception.message,
        "details": exception.details,
    }
    
    if request_id:
        response["request_id"] = request_id
        
    return response


def is_retryable_error(exception: Exception) -> bool:
    """Check if an error is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_errors = (
        ConnectionError,
        CacheConnectionError,
        VectorStoreError,
        LLMError,
    )
    
    return isinstance(exception, retryable_errors)
