"""Constants and enums for the multi-modal RAG system."""

from enum import Enum
from typing import Set


class Modality(str, Enum):
    """Supported data modalities."""
    
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    TABLE = "table"


class DocumentType(str, Enum):
    """Supported document types."""
    
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    PY = "py"
    JS = "js"
    JAVA = "java"
    CPP = "cpp"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"


class QueryIntent(str, Enum):
    """Query intent classification."""
    
    CODE_SEARCH = "code_search"
    VISUAL_SEARCH = "visual_search"
    FACTUAL_SEARCH = "factual_search"
    DEBUG_SEARCH = "debug_search"
    GENERAL_SEARCH = "general_search"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    FAISS = "faiss"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# File processing constants
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
EMBEDDING_BATCH_SIZE = 32

# Allowed file extensions
ALLOWED_EXTENSIONS: Set[str] = {
    ".pdf", ".txt", ".md", ".py", ".js", ".java", ".cpp",
    ".jpg", ".jpeg", ".png", ".docx"
}

# Model constants
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CODE_MODEL = "microsoft/codebert-base"
DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TABLE_MODEL = "google/tapas-base"

# Search constants
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
MIN_SIMILARITY_THRESHOLD = 0.7
RERANK_TOP_K = 20
DIVERSITY_THRESHOLD = 0.8
MIN_RESULTS = 3

# Token limits
MAX_CONTEXT_TOKENS = 8000
MAX_OUTPUT_TOKENS = 2000

# Cache TTL (seconds)
CACHE_TTL_EMBEDDINGS = 3600  # 1 hour
CACHE_TTL_SEARCH = 300       # 5 minutes
CACHE_TTL_LLM = 1800         # 30 minutes

# Rate limiting
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000
RATE_LIMIT_PER_DAY = 10000

# API constants
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Monitoring
METRICS_NAMESPACE = "multimodal_rag"
HEALTH_CHECK_TIMEOUT = 5.0

# OCR settings
OCR_DPI = 300
OCR_LANGUAGES = ["eng"]

# Processing timeouts
ASYNC_TIMEOUT = 30
CONNECTION_TIMEOUT = 10
READ_TIMEOUT = 30

# Embedding dimensions (typical values)
EMBEDDING_DIMENSIONS = {
    Modality.TEXT: 384,    # all-MiniLM-L6-v2
    Modality.CODE: 768,    # CodeBERT
    Modality.IMAGE: 512,   # CLIP
    Modality.TABLE: 768,   # TAPAS
}

# Supported programming languages for code analysis
SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
}

# HTTP status codes for custom responses
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_503_SERVICE_UNAVAILABLE = 503
