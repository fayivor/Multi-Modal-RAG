"""Code parser for analyzing source code files using tree-sitter."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiofiles
import tree_sitter

from ..core.constants import Modality, SUPPORTED_LANGUAGES
from ..core.exceptions import CodeAnalysisError, UnsupportedFileTypeError
from .base import BaseParser, Document

logger = logging.getLogger(__name__)


class CodeParser(BaseParser):
    """Parser for source code files with syntax analysis."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize the code parser.
        
        Args:
            chunk_size: Maximum size of text chunks in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self._parsers: Dict[str, tree_sitter.Parser] = {}
        self._languages: Dict[str, tree_sitter.Language] = {}
    
    def _get_modality(self) -> Modality:
        """Get the modality type for code parser."""
        return Modality.CODE
    
    def validate(self, file_path: Path) -> bool:
        """Validate if the file is a supported code file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a supported code file, False otherwise
        """
        return file_path.suffix.lower() in SUPPORTED_LANGUAGES
    
    async def parse(self, file_path: Path) -> List[Document]:
        """Parse a source code file.
        
        Args:
            file_path: Path to the source code file
            
        Returns:
            List of parsed documents
            
        Raises:
            CodeAnalysisError: If parsing fails
        """
        if not self.validate(file_path):
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                logger.warning(f"Empty code file: {file_path}")
                return []
            
            # Run code analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None, self._analyze_code_sync, file_path, content
            )
            return documents
            
        except UnicodeDecodeError as e:
            raise CodeAnalysisError(f"Failed to decode code file {file_path}: {e}")
        except Exception as e:
            raise CodeAnalysisError(f"Failed to parse code file {file_path}: {e}")
    
    def _analyze_code_sync(self, file_path: Path, content: str) -> List[Document]:
        """Synchronous code analysis method.
        
        Args:
            file_path: Path to the source code file
            content: File content
            
        Returns:
            List of parsed documents
        """
        extension = file_path.suffix.lower()
        language_name = SUPPORTED_LANGUAGES.get(extension)
        
        if not language_name:
            raise CodeAnalysisError(f"Unsupported language for {extension}")
        
        base_metadata = self._create_base_metadata(file_path)
        base_metadata.update({
            "content_type": "text/plain",
            "language": language_name,
            "line_count": len(content.splitlines()),
            "character_count": len(content),
        })
        
        documents = []
        
        try:
            # Try to parse with tree-sitter if available
            parser = self._get_parser(language_name)
            if parser:
                tree = parser.parse(content.encode('utf-8'))
                functions, classes = self._extract_code_elements(tree, content)
                
                base_metadata.update({
                    "functions": [f["name"] for f in functions],
                    "classes": [c["name"] for c in classes],
                    "function_count": len(functions),
                    "class_count": len(classes),
                })
                
                # Create documents for individual functions and classes
                for func in functions:
                    func_metadata = base_metadata.copy()
                    func_metadata.update({
                        "element_type": "function",
                        "element_name": func["name"],
                        "start_line": func["start_line"],
                        "end_line": func["end_line"],
                    })
                    
                    documents.append(Document(
                        content=func["content"],
                        metadata=func_metadata,
                        modality=Modality.CODE
                    ))
                
                for cls in classes:
                    cls_metadata = base_metadata.copy()
                    cls_metadata.update({
                        "element_type": "class",
                        "element_name": cls["name"],
                        "start_line": cls["start_line"],
                        "end_line": cls["end_line"],
                    })
                    
                    documents.append(Document(
                        content=cls["content"],
                        metadata=cls_metadata,
                        modality=Modality.CODE
                    ))
            
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
        
        # Fallback: chunk the entire file
        if not documents:
            documents = self._chunk_text(content, base_metadata)
        
        return documents
    
    def _get_parser(self, language_name: str) -> Optional[tree_sitter.Parser]:
        """Get or create a tree-sitter parser for the language.
        
        Args:
            language_name: Name of the programming language
            
        Returns:
            Tree-sitter parser or None if not available
        """
        if language_name in self._parsers:
            return self._parsers[language_name]
        
        try:
            # This would require tree-sitter language libraries to be installed
            # For now, we'll return None and fall back to simple text chunking
            # In a real implementation, you would load the appropriate language library
            logger.debug(f"Tree-sitter parser not available for {language_name}")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load tree-sitter parser for {language_name}: {e}")
            return None
    
    def _extract_code_elements(
        self, tree: tree_sitter.Tree, content: str
    ) -> tuple[List[Dict], List[Dict]]:
        """Extract functions and classes from the syntax tree.
        
        Args:
            tree: Parsed syntax tree
            content: Original file content
            
        Returns:
            Tuple of (functions, classes) lists
        """
        functions = []
        classes = []
        
        def traverse_node(node):
            """Recursively traverse the syntax tree."""
            if node.type == "function_definition":
                func_info = self._extract_function_info(node, content)
                if func_info:
                    functions.append(func_info)
            elif node.type == "class_definition":
                class_info = self._extract_class_info(node, content)
                if class_info:
                    classes.append(class_info)
            
            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)
        return functions, classes
    
    def _extract_function_info(self, node: tree_sitter.Node, content: str) -> Optional[Dict]:
        """Extract information about a function.
        
        Args:
            node: Function definition node
            content: Original file content
            
        Returns:
            Function information dictionary or None
        """
        try:
            start_byte = node.start_byte
            end_byte = node.end_byte
            func_content = content[start_byte:end_byte]
            
            # Extract function name (simplified)
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            if not name_node:
                return None
            
            func_name = content[name_node.start_byte:name_node.end_byte]
            
            return {
                "name": func_name,
                "content": func_content,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "start_byte": start_byte,
                "end_byte": end_byte,
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract function info: {e}")
            return None
    
    def _extract_class_info(self, node: tree_sitter.Node, content: str) -> Optional[Dict]:
        """Extract information about a class.
        
        Args:
            node: Class definition node
            content: Original file content
            
        Returns:
            Class information dictionary or None
        """
        try:
            start_byte = node.start_byte
            end_byte = node.end_byte
            class_content = content[start_byte:end_byte]
            
            # Extract class name (simplified)
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            if not name_node:
                return None
            
            class_name = content[name_node.start_byte:name_node.end_byte]
            
            return {
                "name": class_name,
                "content": class_content,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "start_byte": start_byte,
                "end_byte": end_byte,
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract class info: {e}")
            return None


class CodeParserFactory:
    """Factory for creating code parsers."""
    
    @classmethod
    def get_parser(cls, file_path: Path, **kwargs) -> Optional[CodeParser]:
        """Get a code parser for a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for parser initialization
            
        Returns:
            CodeParser instance or None if not supported
        """
        if file_path.suffix.lower() in SUPPORTED_LANGUAGES:
            return CodeParser(**kwargs)
        return None
    
    @classmethod
    def supports_file(cls, file_path: Path) -> bool:
        """Check if a file is supported by the code parser.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        return file_path.suffix.lower() in SUPPORTED_LANGUAGES
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(SUPPORTED_LANGUAGES.keys())
