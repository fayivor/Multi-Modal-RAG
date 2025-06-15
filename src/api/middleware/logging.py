"""Logging middleware for request/response logging."""

import logging
import time
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        """Initialize logging middleware.
        
        Args:
            app: FastAPI application
            log_requests: Whether to log incoming requests
            log_responses: Whether to log outgoing responses
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        start_time = time.time()
        
        # Get request ID
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log incoming request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log outgoing response
            if self.log_responses:
                await self._log_response(request, response, process_time, request_id)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            
            logger.error(
                "Request processing failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                error_type=type(e).__name__,
                process_time=process_time,
                request_id=request_id,
                exc_info=True,
            )
            
            raise
    
    async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request details.
        
        Args:
            request: Incoming request
            request_id: Request identifier
        """
        # Get client information
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get request size
        content_length = request.headers.get("content-length")
        request_size = int(content_length) if content_length else 0
        
        # Log request
        logger.info(
            "Incoming request",
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=client_host,
            user_agent=user_agent,
            request_size=request_size,
            content_type=request.headers.get("content-type"),
            request_id=request_id,
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        process_time: float,
        request_id: str
    ) -> None:
        """Log outgoing response details.
        
        Args:
            request: Original request
            response: Outgoing response
            process_time: Request processing time
            request_id: Request identifier
        """
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            response_size = len(response.body) if response.body else 0
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        # Log response
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "response_size": response_size,
            "process_time": process_time,
            "request_id": request_id,
        }
        
        # Add response headers if they contain useful information
        if "x-process-time" in response.headers:
            log_data["x_process_time"] = response.headers["x-process-time"]
        
        if log_level == "error":
            logger.error("Request completed with error", **log_data)
        elif log_level == "warning":
            logger.warning("Request completed with client error", **log_data)
        else:
            logger.info("Request completed successfully", **log_data)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for adding request context to logs."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request context to structured logging.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Get request ID
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Bind request context to logger
        bound_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown",
        )
        
        # Store bound logger in request state
        request.state.logger = bound_logger
        
        return await call_next(request)


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security-related logging."""
    
    def __init__(self, app, log_auth_attempts: bool = True):
        """Initialize security logging middleware.
        
        Args:
            app: FastAPI application
            log_auth_attempts: Whether to log authentication attempts
        """
        super().__init__(app)
        self.log_auth_attempts = log_auth_attempts
        self.suspicious_patterns = [
            "script",
            "javascript:",
            "vbscript:",
            "onload",
            "onerror",
            "eval(",
            "document.cookie",
            "document.write",
            "<script",
            "</script>",
            "SELECT * FROM",
            "DROP TABLE",
            "INSERT INTO",
            "DELETE FROM",
            "UNION SELECT",
            "../",
            "..\\",
            "/etc/passwd",
            "/etc/shadow",
            "cmd.exe",
            "powershell",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security logging.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Check for suspicious patterns
        await self._check_suspicious_patterns(request, request_id)
        
        # Check for rate limiting headers
        await self._check_rate_limiting(request, request_id)
        
        # Process request
        response = await call_next(request)
        
        # Log authentication failures
        if self.log_auth_attempts and response.status_code in [401, 403]:
            await self._log_auth_failure(request, response, request_id)
        
        return response
    
    async def _check_suspicious_patterns(self, request: Request, request_id: str) -> None:
        """Check request for suspicious patterns.
        
        Args:
            request: Incoming request
            request_id: Request identifier
        """
        # Check URL path
        url_path = str(request.url).lower()
        
        # Check query parameters
        query_string = str(request.query_params).lower()
        
        # Check headers
        headers_string = " ".join([
            f"{k}:{v}" for k, v in request.headers.items()
        ]).lower()
        
        # Combine all text to check
        text_to_check = f"{url_path} {query_string} {headers_string}"
        
        # Look for suspicious patterns
        found_patterns = []
        for pattern in self.suspicious_patterns:
            if pattern.lower() in text_to_check:
                found_patterns.append(pattern)
        
        if found_patterns:
            logger.warning(
                "Suspicious request patterns detected",
                patterns=found_patterns,
                url=str(request.url),
                method=request.method,
                client_host=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown"),
                request_id=request_id,
            )
    
    async def _check_rate_limiting(self, request: Request, request_id: str) -> None:
        """Check for rate limiting indicators.
        
        Args:
            request: Incoming request
            request_id: Request identifier
        """
        # Check for rapid requests from same IP
        client_host = request.client.host if request.client else "unknown"
        
        # This is a simplified check - in production you'd use Redis or similar
        # to track request rates per IP
        
        # Log high-frequency requests
        if hasattr(request.state, 'rate_limit_exceeded'):
            logger.warning(
                "Rate limit exceeded",
                client_host=client_host,
                url=str(request.url),
                method=request.method,
                request_id=request_id,
            )
    
    async def _log_auth_failure(
        self,
        request: Request,
        response: Response,
        request_id: str
    ) -> None:
        """Log authentication failures.
        
        Args:
            request: Original request
            response: Response with auth failure
            request_id: Request identifier
        """
        logger.warning(
            "Authentication failure",
            status_code=response.status_code,
            url=str(request.url),
            method=request.method,
            client_host=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            authorization_header_present="authorization" in request.headers,
            request_id=request_id,
        )
