"""Code embedding encoder using CodeBERT and similar models."""

import logging
import re
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..core.constants import Modality, SUPPORTED_LANGUAGES
from ..core.exceptions import EncodingError, ModelLoadError
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class CodeEncoder(BaseEncoder):
    """Code embedding encoder using transformer models like CodeBERT."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str = "cpu",
        batch_size: int = 16,  # Smaller batch size for code
        normalize: bool = True,
        max_seq_length: int = 512,
    ) -> None:
        """Initialize the code encoder.
        
        Args:
            model_name: Name of the code transformer model
            device: Device to run the model on
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            max_seq_length: Maximum sequence length for tokenization
        """
        super().__init__(model_name, device, batch_size, normalize)
        self.max_seq_length = max_seq_length
    
    async def load_model(self) -> None:
        """Load the code transformer model."""
        try:
            logger.info(f"Loading code encoder: {self.model_name}")
            
            # Load model and tokenizer in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            # Move model to device
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            
            # Set model to evaluation mode
            self._model.eval()
            
            self._is_loaded = True
            logger.info(f"Successfully loaded code encoder: {self.model_name}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load code encoder {self.model_name}: {e}")
    
    def _load_model_sync(self) -> tuple:
        """Synchronously load the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        return model, tokenizer
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        language: str = None,
        include_comments: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode code inputs into embeddings.
        
        Args:
            inputs: Code string or list of code strings
            language: Programming language (auto-detected if None)
            include_comments: Whether to include comments in encoding
            **kwargs: Additional encoding parameters
            
        Returns:
            Numpy array of embeddings
        """
        await self.ensure_loaded()
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if not inputs:
            return np.array([])
        
        try:
            # Preprocess code inputs
            processed_inputs = []
            for code in inputs:
                processed_code = self._preprocess_code(
                    code, language=language, include_comments=include_comments
                )
                processed_inputs.append(processed_code)
            
            # Process in batches
            if len(processed_inputs) <= self.batch_size:
                embeddings = await self._encode_batch(processed_inputs, **kwargs)
            else:
                all_embeddings = []
                batches = self._batch_inputs(processed_inputs)
                
                for batch in batches:
                    batch_embeddings = await self._encode_batch(batch, **kwargs)
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode code inputs: {e}")
    
    def _encode_batch_sync(self, batch: List[str], kwargs: Dict) -> np.ndarray:
        """Synchronous batch encoding method.
        
        Args:
            batch: Batch of code strings to encode
            kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        # Tokenize the batch
        inputs = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            
            # Use [CLS] token embedding or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def _preprocess_code(
        self,
        code: str,
        language: str = None,
        include_comments: bool = True
    ) -> str:
        """Preprocess code for better embedding quality.
        
        Args:
            code: Raw code string
            language: Programming language
            include_comments: Whether to include comments
            
        Returns:
            Preprocessed code string
        """
        if not include_comments:
            code = self._remove_comments(code, language)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code.strip())
        
        # Truncate if too long (rough token estimation)
        max_chars = self.max_seq_length * 4  # Rough estimate
        if len(code) > max_chars:
            code = code[:max_chars]
        
        return code
    
    def _remove_comments(self, code: str, language: str = None) -> str:
        """Remove comments from code.
        
        Args:
            code: Code string
            language: Programming language
            
        Returns:
            Code without comments
        """
        if language is None:
            language = self._detect_language(code)
        
        # Simple comment removal patterns
        if language in ["python", "ruby", "shell"]:
            # Remove # comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        elif language in ["javascript", "java", "cpp", "c", "cs", "go", "rust"]:
            # Remove // comments
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
            # Remove /* */ comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code content.
        
        Args:
            code: Code string
            
        Returns:
            Detected language name
        """
        # Simple heuristic-based language detection
        if 'def ' in code and 'import ' in code:
            return "python"
        elif 'function ' in code and 'var ' in code:
            return "javascript"
        elif 'public class ' in code and 'import java' in code:
            return "java"
        elif '#include' in code and 'int main' in code:
            return "cpp"
        elif 'func ' in code and 'package ' in code:
            return "go"
        else:
            return "unknown"
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return default dimensions for common models
            model_dims = {
                "microsoft/codebert-base": 768,
                "microsoft/graphcodebert-base": 768,
                "microsoft/unixcoder-base": 768,
                "huggingface/CodeBERTa-small-v1": 768,
            }
            return model_dims.get(self.model_name, 768)
        
        return self._model.config.hidden_size
    
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles.
        
        Returns:
            Code modality
        """
        return Modality.CODE
    
    def encode_function(
        self,
        function_code: str,
        function_name: str = None,
        docstring: str = None,
        **kwargs
    ) -> np.ndarray:
        """Encode a function with its metadata.
        
        Args:
            function_code: Function source code
            function_name: Function name
            docstring: Function docstring
            **kwargs: Additional encoding parameters
            
        Returns:
            Function embedding
        """
        # Combine function components
        components = []
        
        if function_name:
            components.append(f"Function: {function_name}")
        
        if docstring:
            components.append(f"Description: {docstring}")
        
        components.append(function_code)
        
        combined_text = "\n".join(components)
        return self.encode(combined_text, **kwargs)
    
    def encode_class(
        self,
        class_code: str,
        class_name: str = None,
        docstring: str = None,
        **kwargs
    ) -> np.ndarray:
        """Encode a class with its metadata.
        
        Args:
            class_code: Class source code
            class_name: Class name
            docstring: Class docstring
            **kwargs: Additional encoding parameters
            
        Returns:
            Class embedding
        """
        # Combine class components
        components = []
        
        if class_name:
            components.append(f"Class: {class_name}")
        
        if docstring:
            components.append(f"Description: {docstring}")
        
        components.append(class_code)
        
        combined_text = "\n".join(components)
        return self.encode(combined_text, **kwargs)


class LanguageSpecificCodeEncoder(CodeEncoder):
    """Language-specific code encoder for better performance on specific languages."""
    
    def __init__(
        self,
        language: str,
        model_name: str = None,
        **kwargs
    ) -> None:
        """Initialize the language-specific code encoder.
        
        Args:
            language: Programming language
            model_name: Model name (auto-selected if None)
            **kwargs: Additional arguments for base encoder
        """
        self.language = language
        
        if model_name is None:
            model_name = self._get_language_model(language)
        
        super().__init__(model_name=model_name, **kwargs)
    
    def _get_language_model(self, language: str) -> str:
        """Get the appropriate model for a programming language.
        
        Args:
            language: Programming language
            
        Returns:
            Model name for the language
        """
        # This would be expanded with language-specific models
        language_models = {
            "python": "microsoft/codebert-base",
            "java": "microsoft/codebert-base",
            "javascript": "microsoft/codebert-base",
            "cpp": "microsoft/codebert-base",
            "go": "microsoft/codebert-base",
        }
        
        return language_models.get(language, "microsoft/codebert-base")
    
    def get_language(self) -> str:
        """Get the programming language this encoder is specialized for.
        
        Returns:
            Programming language name
        """
        return self.language
