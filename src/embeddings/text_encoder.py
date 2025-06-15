"""Text embedding encoder using sentence transformers."""

import logging
from typing import Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.constants import Modality
from ..core.exceptions import EncodingError, ModelLoadError
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class TextEncoder(BaseEncoder):
    """Text embedding encoder using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int = 512,
    ) -> None:
        """Initialize the text encoder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            max_seq_length: Maximum sequence length for tokenization
        """
        super().__init__(model_name, device, batch_size, normalize)
        self.max_seq_length = max_seq_length
    
    async def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading text encoder: {self.model_name}")
            
            # Load model in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            # Set device
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            
            # Set max sequence length
            if hasattr(self._model, 'max_seq_length'):
                self._model.max_seq_length = self.max_seq_length
            
            self._is_loaded = True
            logger.info(f"Successfully loaded text encoder: {self.model_name}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load text encoder {self.model_name}: {e}")
    
    def _load_model_sync(self) -> SentenceTransformer:
        """Synchronously load the model."""
        return SentenceTransformer(self.model_name, device=self.device)
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode text inputs into embeddings.
        
        Args:
            inputs: Text string or list of text strings
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
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
            # Process in batches for large inputs
            if len(inputs) <= self.batch_size:
                embeddings = await self._encode_batch(
                    inputs, show_progress_bar=show_progress_bar, **kwargs
                )
            else:
                # Process in multiple batches
                all_embeddings = []
                batches = self._batch_inputs(inputs)
                
                for i, batch in enumerate(batches):
                    batch_embeddings = await self._encode_batch(
                        batch,
                        show_progress_bar=show_progress_bar and i == 0,
                        **kwargs
                    )
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode text inputs: {e}")
    
    def _encode_batch_sync(self, batch: List[str], kwargs: Dict) -> np.ndarray:
        """Synchronous batch encoding method.
        
        Args:
            batch: Batch of text strings to encode
            kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        return self._model.encode(
            batch,
            convert_to_numpy=True,
            **kwargs
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return default dimension for common models
            model_dims = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/all-distilroberta-v1": 768,
                "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
                "sentence-transformers/paraphrase-mpnet-base-v2": 768,
            }
            return model_dims.get(self.model_name, 384)
        
        return self._model.get_sentence_embedding_dimension()
    
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles.
        
        Returns:
            Text modality
        """
        return Modality.TEXT
    
    def get_max_sequence_length(self) -> int:
        """Get the maximum sequence length.
        
        Returns:
            Maximum sequence length
        """
        if not self._is_loaded:
            return self.max_seq_length
        
        return getattr(self._model, 'max_seq_length', self.max_seq_length)
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to fit within model limits.
        
        Args:
            text: Input text
            max_length: Maximum length (default: model's max_seq_length)
            
        Returns:
            Truncated text
        """
        if max_length is None:
            max_length = self.get_max_sequence_length()
        
        # Simple word-based truncation
        words = text.split()
        if len(words) <= max_length:
            return text
        
        return " ".join(words[:max_length])
    
    def encode_with_metadata(
        self,
        texts: List[str],
        metadata: List[Dict] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Encode texts and return embeddings with metadata.
        
        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary with embeddings and metadata
        """
        embeddings = self.encode(texts, **kwargs)
        
        result = {
            "embeddings": embeddings,
            "texts": texts,
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
        }
        
        if metadata:
            result["metadata"] = metadata
        
        return result


class MultilingualTextEncoder(TextEncoder):
    """Multilingual text encoder for cross-language embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        **kwargs
    ) -> None:
        """Initialize the multilingual text encoder.
        
        Args:
            model_name: Name of the multilingual model
            **kwargs: Additional arguments for base encoder
        """
        super().__init__(model_name=model_name, **kwargs)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        # This is a simplified list for common multilingual models
        return [
            "en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko",
            "ar", "hi", "th", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ]


class DomainSpecificTextEncoder(TextEncoder):
    """Domain-specific text encoder for specialized domains."""
    
    def __init__(
        self,
        domain: str,
        model_name: str = None,
        **kwargs
    ) -> None:
        """Initialize the domain-specific text encoder.
        
        Args:
            domain: Domain name (e.g., 'biomedical', 'legal', 'scientific')
            model_name: Model name (auto-selected if None)
            **kwargs: Additional arguments for base encoder
        """
        self.domain = domain
        
        if model_name is None:
            model_name = self._get_domain_model(domain)
        
        super().__init__(model_name=model_name, **kwargs)
    
    def _get_domain_model(self, domain: str) -> str:
        """Get the appropriate model for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Model name for the domain
        """
        domain_models = {
            "biomedical": "sentence-transformers/all-MiniLM-L6-v2",  # Placeholder
            "legal": "sentence-transformers/all-MiniLM-L6-v2",      # Placeholder
            "scientific": "sentence-transformers/all-MiniLM-L6-v2", # Placeholder
            "technical": "sentence-transformers/all-MiniLM-L6-v2",  # Placeholder
        }
        
        return domain_models.get(domain, "sentence-transformers/all-MiniLM-L6-v2")
    
    def get_domain(self) -> str:
        """Get the domain this encoder is specialized for.
        
        Returns:
            Domain name
        """
        return self.domain
