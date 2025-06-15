"""Document parsers for PDF, DOCX, and text files."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import aiofiles
from docx import Document as DocxDocument
from pypdf import PdfReader

from ..core.constants import Modality
from ..core.exceptions import DocumentParsingError, UnsupportedFileTypeError
from .base import BaseParser, Document

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """Parser for plain text files."""
    
    def _get_modality(self) -> Modality:
        """Get the modality type for text parser."""
        return Modality.TEXT
    
    def validate(self, file_path: Path) -> bool:
        """Validate if the file is a text file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a text file, False otherwise
        """
        return file_path.suffix.lower() in {".txt", ".md"}
    
    async def parse(self, file_path: Path) -> List[Document]:
        """Parse a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of parsed documents
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        if not self.validate(file_path):
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return []
            
            metadata = self._create_base_metadata(file_path)
            metadata.update({
                "content_type": "text/plain",
                "encoding": "utf-8",
                "line_count": len(content.splitlines()),
                "character_count": len(content),
            })
            
            return self._chunk_text(content, metadata)
            
        except UnicodeDecodeError as e:
            raise DocumentParsingError(f"Failed to decode text file {file_path}: {e}")
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse text file {file_path}: {e}")


class PDFParser(BaseParser):
    """Parser for PDF files."""
    
    def _get_modality(self) -> Modality:
        """Get the modality type for PDF parser."""
        return Modality.TEXT
    
    def validate(self, file_path: Path) -> bool:
        """Validate if the file is a PDF.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a PDF, False otherwise
        """
        return file_path.suffix.lower() == ".pdf"
    
    async def parse(self, file_path: Path) -> List[Document]:
        """Parse a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of parsed documents
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        if not self.validate(file_path):
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            # Run PDF parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(None, self._parse_pdf_sync, file_path)
            return documents
            
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF file {file_path}: {e}")
    
    def _parse_pdf_sync(self, file_path: Path) -> List[Document]:
        """Synchronous PDF parsing method.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of parsed documents
        """
        documents = []
        
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            
            base_metadata = self._create_base_metadata(file_path)
            base_metadata.update({
                "content_type": "application/pdf",
                "total_pages": len(reader.pages),
                "pdf_metadata": reader.metadata._data if reader.metadata else {},
            })
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    
                    if not text.strip():
                        logger.warning(f"Empty page {page_num + 1} in {file_path}")
                        continue
                    
                    page_metadata = base_metadata.copy()
                    page_metadata.update({
                        "page_number": page_num + 1,
                        "character_count": len(text),
                    })
                    
                    # Chunk the page content
                    page_chunks = self._chunk_text(text, page_metadata)
                    documents.extend(page_chunks)
                    
                except Exception as e:
                    logger.error(f"Failed to extract text from page {page_num + 1} in {file_path}: {e}")
                    continue
        
        return documents


class DOCXParser(BaseParser):
    """Parser for DOCX files."""
    
    def _get_modality(self) -> Modality:
        """Get the modality type for DOCX parser."""
        return Modality.TEXT
    
    def validate(self, file_path: Path) -> bool:
        """Validate if the file is a DOCX file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a DOCX file, False otherwise
        """
        return file_path.suffix.lower() == ".docx"
    
    async def parse(self, file_path: Path) -> List[Document]:
        """Parse a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of parsed documents
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        if not self.validate(file_path):
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            # Run DOCX parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(None, self._parse_docx_sync, file_path)
            return documents
            
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse DOCX file {file_path}: {e}")
    
    def _parse_docx_sync(self, file_path: Path) -> List[Document]:
        """Synchronous DOCX parsing method.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of parsed documents
        """
        doc = DocxDocument(file_path)
        
        # Extract text from paragraphs
        paragraphs = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)
        
        # Combine all paragraphs
        content = "\n\n".join(paragraphs)
        
        if not content.strip():
            logger.warning(f"Empty DOCX file: {file_path}")
            return []
        
        metadata = self._create_base_metadata(file_path)
        metadata.update({
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "paragraph_count": len(paragraphs),
            "character_count": len(content),
        })
        
        return self._chunk_text(content, metadata)


class DocumentParserFactory:
    """Factory for creating document parsers."""
    
    _parsers = {
        ".txt": TextParser,
        ".md": TextParser,
        ".pdf": PDFParser,
        ".docx": DOCXParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: Path, **kwargs) -> Optional[BaseParser]:
        """Get the appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for parser initialization
            
        Returns:
            Parser instance or None if no parser is available
        """
        extension = file_path.suffix.lower()
        parser_class = cls._parsers.get(extension)
        
        if parser_class:
            return parser_class(**kwargs)
        
        return None
    
    @classmethod
    def supports_file(cls, file_path: Path) -> bool:
        """Check if a file is supported by any parser.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        extension = file_path.suffix.lower()
        return extension in cls._parsers
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(cls._parsers.keys())
