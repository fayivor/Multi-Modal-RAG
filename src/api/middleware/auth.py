"""Authentication middleware for API security."""

import logging
from typing import Callable, Optional

import structlog
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.config import settings

logger = structlog.get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple authentication middleware."""
    
    def __init__(self, app):
        """Initialize authentication middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/health/",
            "/health/live",
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
        }
        
        # API key for simple authentication
        self.api_key = getattr(settings, 'api_key', None)
        
        # Enable/disable authentication
        self.enabled = getattr(settings, 'enable_auth', False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Skip authentication if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Authenticate request
        try:
            await self._authenticate_request(request)
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Authentication error",
                error=str(e),
                path=request.url.path,
                method=request.method,
            )
            raise HTTPException(status_code=500, detail="Authentication error")
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public.
        
        Args:
            path: Request path
            
        Returns:
            True if endpoint is public
        """
        # Exact match
        if path in self.public_endpoints:
            return True
        
        # Pattern matching for health endpoints
        if path.startswith("/health"):
            return True
        
        return False
    
    async def _authenticate_request(self, request: Request) -> None:
        """Authenticate the request.
        
        Args:
            request: Incoming request
            
        Raises:
            HTTPException: If authentication fails
        """
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            # Check for API key in query parameter
            api_key = request.query_params.get("api_key")
        
        if not api_key:
            # Check for Bearer token
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                api_key = authorization[7:]  # Remove "Bearer " prefix
        
        if not api_key:
            logger.warning(
                "Missing API key",
                path=request.url.path,
                method=request.method,
                client_host=request.client.host if request.client else "unknown",
            )
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate API key
        if not self._validate_api_key(api_key):
            logger.warning(
                "Invalid API key",
                path=request.url.path,
                method=request.method,
                client_host=request.client.host if request.client else "unknown",
                api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else api_key,
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Store authentication info in request state
        request.state.authenticated = True
        request.state.api_key = api_key
        
        logger.info(
            "Request authenticated",
            path=request.url.path,
            method=request.method,
            api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else api_key,
        )
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key is valid
        """
        # Simple validation against configured API key
        if self.api_key and api_key == self.api_key:
            return True
        
        # In production, you might validate against a database
        # or external authentication service
        
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = getattr(settings, 'enable_rate_limiting', True)
        
        # Simple in-memory rate limiting
        # In production, use Redis or similar
        self.request_counts = {}
        self.last_reset = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        if not self.enabled:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                }
            )
        
        # Increment request count
        self._increment_request_count(client_ip)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if rate limited
        """
        import time
        
        current_time = time.time()
        
        # Reset counter if more than a minute has passed
        if client_ip in self.last_reset:
            if current_time - self.last_reset[client_ip] > 60:
                self.request_counts[client_ip] = 0
                self.last_reset[client_ip] = current_time
        else:
            self.last_reset[client_ip] = current_time
            self.request_counts[client_ip] = 0
        
        # Check if limit exceeded
        current_count = self.request_counts.get(client_ip, 0)
        return current_count >= self.requests_per_minute
    
    def _increment_request_count(self, client_ip: str) -> None:
        """Increment request count for client.
        
        Args:
            client_ip: Client IP address
        """
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = 0
        
        self.request_counts[client_ip] += 1
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Number of remaining requests
        """
        current_count = self.request_counts.get(client_ip, 0)
        return max(0, self.requests_per_minute - current_count)


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS security middleware."""
    
    def __init__(self, app):
        """Initialize CORS security middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.allowed_origins = getattr(settings, 'cors_origins', [])
        self.enabled = getattr(settings, 'enable_cors_security', True)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with CORS security.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        if not self.enabled:
            return await call_next(request)
        
        # Check origin for CORS requests
        origin = request.headers.get("Origin")
        
        if origin and not self._is_allowed_origin(origin):
            logger.warning(
                "Blocked request from unauthorized origin",
                origin=origin,
                path=request.url.path,
                method=request.method,
            )
            raise HTTPException(
                status_code=403,
                detail="Origin not allowed"
            )
        
        return await call_next(request)
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed.
        
        Args:
            origin: Request origin
            
        Returns:
            True if origin is allowed
        """
        if not self.allowed_origins:
            return True  # Allow all if no restrictions configured
        
        return origin in self.allowed_origins
