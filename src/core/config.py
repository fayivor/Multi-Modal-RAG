"""Configuration management using Pydantic settings."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

from .constants import LLMProvider, VectorStoreType


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="sqlite:///./multi_modal_rag.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""
    
    type: VectorStoreType = Field(default=VectorStoreType.QDRANT, env="VECTOR_DB_TYPE")
    
    # Qdrant settings
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_timeout: int = Field(default=30, env="QDRANT_TIMEOUT")
    qdrant_pool_connections: int = Field(default=10, env="QDRANT_POOL_CONNECTIONS")
    
    # Weaviate settings
    weaviate_host: str = Field(default="localhost", env="WEAVIATE_HOST")
    weaviate_port: int = Field(default=8080, env="WEAVIATE_PORT")
    weaviate_scheme: str = Field(default="http", env="WEAVIATE_SCHEME")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")


class LLMSettings(BaseSettings):
    """LLM configuration settings."""
    
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=2000, env="ANTHROPIC_MAX_TOKENS")
    
    # Local model settings
    local_model_path: Optional[str] = Field(default=None, env="LOCAL_MODEL_PATH")
    local_model_device: str = Field(default="cpu", env="LOCAL_MODEL_DEVICE")
    
    @validator("openai_api_key")
    def validate_openai_key(cls, v, values):
        """Validate OpenAI API key when using OpenAI provider."""
        if values.get("provider") == LLMProvider.OPENAI and not v:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        return v
    
    @validator("anthropic_api_key")
    def validate_anthropic_key(cls, v, values):
        """Validate Anthropic API key when using Anthropic provider."""
        if values.get("provider") == LLMProvider.ANTHROPIC and not v:
            raise ValueError("Anthropic API key is required when using Anthropic provider")
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration settings."""
    
    text_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="TEXT_EMBEDDING_MODEL"
    )
    code_model: str = Field(
        default="microsoft/codebert-base",
        env="CODE_EMBEDDING_MODEL"
    )
    image_model: str = Field(
        default="openai/clip-vit-base-patch32",
        env="IMAGE_EMBEDDING_MODEL"
    )
    table_model: str = Field(
        default="google/tapas-base",
        env="TABLE_EMBEDDING_MODEL"
    )
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    device: str = Field(default="cpu", env="EMBEDDING_DEVICE")


class ProcessingSettings(BaseSettings):
    """Document processing configuration settings."""
    
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_extensions: str = Field(
        default=".pdf,.txt,.md,.py,.js,.java,.cpp,.jpg,.png,.jpeg,.docx",
        env="ALLOWED_EXTENSIONS"
    )
    upload_dir: Path = Field(default=Path("./uploads"), env="UPLOAD_DIR")
    temp_dir: Path = Field(default=Path("./temp"), env="TEMP_DIR")
    
    @validator("allowed_extensions")
    def parse_extensions(cls, v):
        """Parse allowed extensions from comma-separated string."""
        return set(ext.strip() for ext in v.split(","))


class SearchSettings(BaseSettings):
    """Search configuration settings."""
    
    default_top_k: int = Field(default=10, env="DEFAULT_TOP_K")
    max_top_k: int = Field(default=100, env="MAX_TOP_K")
    min_similarity_threshold: float = Field(default=0.7, env="MIN_SIMILARITY_THRESHOLD")
    rerank_top_k: int = Field(default=20, env="RERANK_TOP_K")
    diversity_threshold: float = Field(default=0.8, env="DIVERSITY_THRESHOLD")


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    ttl_embeddings: int = Field(default=3600, env="CACHE_TTL_EMBEDDINGS")  # 1 hour
    ttl_search: int = Field(default=300, env="CACHE_TTL_SEARCH")  # 5 minutes
    ttl_llm: int = Field(default=1800, env="CACHE_TTL_LLM")  # 30 minutes


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration settings."""
    
    per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    per_day: int = Field(default=10000, env="RATE_LIMIT_PER_DAY")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""
    
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")
    sentry_traces_sample_rate: float = Field(default=0.1, env="SENTRY_TRACES_SAMPLE_RATE")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field(default="multi-modal-rag", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    app_description: str = Field(
        default="Multi-Modal RAG Technical Documentation Assistant",
        env="APP_DESCRIPTION"
    )
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=True, env="RELOAD")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # CORS settings
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Documentation settings
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    redis: RedisSettings = RedisSettings()
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    processing: ProcessingSettings = ProcessingSettings()
    search: SearchSettings = SearchSettings()
    cache: CacheSettings = CacheSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
