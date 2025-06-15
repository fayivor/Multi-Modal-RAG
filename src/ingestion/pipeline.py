"""Ingestion pipeline orchestrator for processing multiple file types."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.config import settings
from ..core.constants import ProcessingStatus
from ..core.exceptions import (
    FileSizeExceededError,
    IngestionError,
    UnsupportedFileTypeError,
    ValidationError,
)
from .base import Document, ProcessingResult
from .code_parser import CodeParserFactory
from .document_parser import DocumentParserFactory
from .image_processor import ImageProcessorFactory

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main ingestion pipeline for processing various file types."""
    
    def __init__(
        self,
        max_file_size: int = None,
        allowed_extensions: set = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> None:
        """Initialize the ingestion pipeline.
        
        Args:
            max_file_size: Maximum allowed file size in bytes
            allowed_extensions: Set of allowed file extensions
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.max_file_size = max_file_size or settings.processing.max_file_size
        self.allowed_extensions = allowed_extensions or settings.processing.allowed_extensions
        self.chunk_size = chunk_size or settings.processing.chunk_size
        self.chunk_overlap = chunk_overlap or settings.processing.chunk_overlap
        
        # Initialize processor factories
        self.document_factory = DocumentParserFactory()
        self.code_factory = CodeParserFactory()
        self.image_factory = ImageProcessorFactory()
    
    async def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processing result with documents and metadata
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        try:
            # Validate file
            self._validate_file(file_path)
            
            # Get appropriate processor
            processor = self._get_processor(file_path)
            if not processor:
                raise UnsupportedFileTypeError(
                    f"No processor available for file type: {file_path.suffix}"
                )
            
            # Process the file
            if hasattr(processor, 'parse'):
                # Document or code parser
                documents = await processor.parse(file_path)
                result = ProcessingResult(
                    documents=documents,
                    status=ProcessingStatus.COMPLETED,
                    processing_time=time.time() - start_time,
                    metadata={
                        "processor_type": type(processor).__name__,
                        "file_path": str(file_path),
                        "document_count": len(documents),
                    }
                )
            else:
                # Image processor
                result = await processor.process(file_path)
                result.processing_time = time.time() - start_time
            
            logger.info(
                f"Successfully processed {file_path}: "
                f"{len(result.documents)} documents in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time,
                metadata={"file_path": str(file_path)}
            )
    
    async def process_files(
        self, file_paths: List[Union[str, Path]], max_concurrent: int = 5
    ) -> List[ProcessingResult]:
        """Process multiple files concurrently.
        
        Args:
            file_paths: List of file paths to process
            max_concurrent: Maximum number of concurrent processing tasks
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_file(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error_message=str(result),
                    metadata={"file_path": str(file_paths[i])}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        max_concurrent: int = 5,
    ) -> List[ProcessingResult]:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories recursively
            max_concurrent: Maximum number of concurrent processing tasks
            
        Returns:
            List of processing results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise ValidationError(f"Directory does not exist: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValidationError(f"Path is not a directory: {directory_path}")
        
        # Find all supported files
        file_paths = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and self._is_supported_file(file_path):
                file_paths.append(file_path)
        
        logger.info(f"Found {len(file_paths)} supported files in {directory_path}")
        
        if not file_paths:
            return []
        
        return await self.process_files(file_paths, max_concurrent)
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate a file before processing.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            ValidationError: If file validation fails
            FileSizeExceededError: If file is too large
            UnsupportedFileTypeError: If file type is not supported
        """
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise FileSizeExceededError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"size ({self.max_file_size} bytes): {file_path}"
            )
        
        # Check file extension
        if not self._is_supported_file(file_path):
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix}"
            )
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is supported by any processor.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is supported, False otherwise
        """
        extension = file_path.suffix.lower()
        return extension in self.allowed_extensions
    
    def _get_processor(self, file_path: Path):
        """Get the appropriate processor for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor instance or None if no processor is available
        """
        # Try document parser first
        parser = self.document_factory.get_parser(
            file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if parser:
            return parser
        
        # Try code parser
        parser = self.code_factory.get_parser(
            file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if parser:
            return parser
        
        # Try image processor
        processor = self.image_factory.get_processor(file_path)
        if processor:
            return processor
        
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        extensions = set()
        extensions.update(self.document_factory.get_supported_extensions())
        extensions.update(self.code_factory.get_supported_extensions())
        extensions.update([".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"])
        return sorted(list(extensions))
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict:
        """Get processing statistics from results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with processing statistics
        """
        total_files = len(results)
        successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        total_documents = sum(len(r.documents) for r in results)
        total_time = sum(r.processing_time or 0 for r in results)
        
        return {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_files if total_files > 0 else 0,
            "total_documents": total_documents,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / total_files if total_files > 0 else 0,
            "documents_per_file": total_documents / successful if successful > 0 else 0,
        }


# Global pipeline instance
pipeline = IngestionPipeline()
