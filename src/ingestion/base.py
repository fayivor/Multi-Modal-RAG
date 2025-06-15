"""Base classes for document ingestion and processing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.constants import Modality, ProcessingStatus
from ..core.exceptions import ValidationError


class Document(BaseModel):
    """Represents a processed document with metadata and content."""
    
    content: str = Field(..., description="The extracted text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata including source, page, type, timestamp"
    )
    modality: Modality = Field(..., description="The type of content modality")
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="Pre-computed embeddings for the document"
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this chunk"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of the parent document this chunk belongs to"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class ProcessingResult(BaseModel):
    """Result of document processing operation."""
    
    documents: List[Document] = Field(
        default_factory=list,
        description="List of processed documents"
    )
    status: ProcessingStatus = Field(
        default=ProcessingStatus.COMPLETED,
        description="Processing status"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Time taken to process in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata"
    )


class BaseParser(ABC):
    """Abstract base class for all document parsers."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize the parser.
        
        Args:
            chunk_size: Maximum size of text chunks in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    async def parse(self, file_path: Path) -> List[Document]:
        """Parse a file and return a list of documents.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of parsed documents
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        pass
    
    @abstractmethod
    def validate(self, file_path: Path) -> bool:
        """Validate if the file can be processed by this parser.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file can be processed, False otherwise
        """
        pass
    
    def _create_base_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create base metadata for a document.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Base metadata dictionary
        """
        import time
        
        return {
            "source": str(file_path),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_extension": file_path.suffix.lower(),
            "timestamp": time.time(),
            "parser": self.__class__.__name__,
        }
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the document
            
        Returns:
            List of document chunks
        """
        # Simple word-based chunking (can be improved with token-based chunking)
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            # Text is small enough to be a single chunk
            return [Document(
                content=text,
                metadata={**metadata, "chunk_index": 0, "total_chunks": 1},
                modality=self._get_modality()
            )]
        
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "start_word": start_idx,
                "end_word": end_idx,
            }
            
            chunks.append(Document(
                content=chunk_text,
                metadata=chunk_metadata,
                modality=self._get_modality()
            ))
            
            # Move start index with overlap
            start_idx = end_idx - self.chunk_overlap
            chunk_index += 1
        
        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    @abstractmethod
    def _get_modality(self) -> Modality:
        """Get the modality type for this parser.
        
        Returns:
            The modality type
        """
        pass


class BaseProcessor(ABC):
    """Abstract base class for specialized processors."""
    
    def __init__(self) -> None:
        """Initialize the processor."""
        pass
    
    @abstractmethod
    async def process(self, file_path: Path) -> ProcessingResult:
        """Process a file and return the result.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processing result with documents and metadata
        """
        pass
    
    @abstractmethod
    def supports_file(self, file_path: Path) -> bool:
        """Check if this processor supports the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is supported, False otherwise
        """
        pass
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate file before processing.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            ValidationError: If file validation fails
        """
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        if file_path.stat().st_size == 0:
            raise ValidationError(f"File is empty: {file_path}")


class BaseExtractor(ABC):
    """Abstract base class for content extractors."""
    
    @abstractmethod
    async def extract(self, data: Any) -> str:
        """Extract text content from data.
        
        Args:
            data: Input data to extract content from
            
        Returns:
            Extracted text content
        """
        pass
    
    @abstractmethod
    def can_extract(self, data: Any) -> bool:
        """Check if this extractor can handle the given data.
        
        Args:
            data: Data to check
            
        Returns:
            True if the data can be extracted, False otherwise
        """
        pass
