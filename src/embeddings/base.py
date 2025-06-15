"""Base classes for embedding encoders."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..core.constants import Modality
from ..core.exceptions import EncodingError, ModelLoadError

logger = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """Abstract base class for all embedding encoders."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        """Initialize the encoder.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on (cpu, cuda, mps)
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the embedding model."""
        pass
    
    @abstractmethod
    async def encode(
        self, inputs: Union[str, List[str]], **kwargs
    ) -> np.ndarray:
        """Encode inputs into embeddings.
        
        Args:
            inputs: Input text(s) or data to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles.
        
        Returns:
            Modality type
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._is_loaded
    
    async def ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self._is_loaded:
            await self.load_model()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings using L2 normalization.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        if not self.normalize:
            return embeddings
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _batch_inputs(self, inputs: List[Any]) -> List[List[Any]]:
        """Split inputs into batches.
        
        Args:
            inputs: List of inputs to batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(inputs), self.batch_size):
            batches.append(inputs[i:i + self.batch_size])
        return batches
    
    async def _encode_batch(self, batch: List[Any], **kwargs) -> np.ndarray:
        """Encode a single batch of inputs.
        
        Args:
            batch: Batch of inputs to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._encode_batch_sync, batch, kwargs
        )
    
    @abstractmethod
    def _encode_batch_sync(self, batch: List[Any], kwargs: Dict) -> np.ndarray:
        """Synchronous batch encoding method.
        
        Args:
            batch: Batch of inputs to encode
            kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "modality": self.get_modality().value,
            "embedding_dimension": self.get_embedding_dimension(),
            "is_loaded": self._is_loaded,
        }


class EmbeddingManager:
    """Manager for multiple embedding encoders with lazy loading."""
    
    def __init__(self) -> None:
        """Initialize the embedding manager."""
        self._encoders: Dict[Modality, BaseEncoder] = {}
        self._encoder_configs: Dict[Modality, Dict[str, Any]] = {}
    
    def register_encoder(
        self,
        modality: Modality,
        encoder_class: type,
        **config
    ) -> None:
        """Register an encoder for a modality.
        
        Args:
            modality: Modality type
            encoder_class: Encoder class
            **config: Configuration for the encoder
        """
        self._encoder_configs[modality] = {
            "encoder_class": encoder_class,
            "config": config
        }
        logger.info(f"Registered encoder for {modality.value}: {encoder_class.__name__}")
    
    async def get_encoder(self, modality: Modality) -> BaseEncoder:
        """Get encoder for a modality, loading it if necessary.
        
        Args:
            modality: Modality type
            
        Returns:
            Encoder instance
            
        Raises:
            ModelLoadError: If encoder is not registered or fails to load
        """
        if modality not in self._encoder_configs:
            raise ModelLoadError(f"No encoder registered for modality: {modality.value}")
        
        if modality not in self._encoders:
            # Lazy load the encoder
            config = self._encoder_configs[modality]
            encoder_class = config["encoder_class"]
            encoder_config = config["config"]
            
            try:
                encoder = encoder_class(**encoder_config)
                await encoder.load_model()
                self._encoders[modality] = encoder
                logger.info(f"Loaded encoder for {modality.value}")
            except Exception as e:
                raise ModelLoadError(f"Failed to load encoder for {modality.value}: {e}")
        
        return self._encoders[modality]
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        modality: Modality,
        **kwargs
    ) -> np.ndarray:
        """Encode inputs using the appropriate encoder.
        
        Args:
            inputs: Input data to encode
            modality: Modality type
            **kwargs: Additional encoding parameters
            
        Returns:
            Embeddings array
        """
        encoder = await self.get_encoder(modality)
        return await encoder.encode(inputs, **kwargs)
    
    def get_registered_modalities(self) -> List[Modality]:
        """Get list of registered modalities.
        
        Returns:
            List of registered modalities
        """
        return list(self._encoder_configs.keys())
    
    def get_loaded_modalities(self) -> List[Modality]:
        """Get list of loaded modalities.
        
        Returns:
            List of loaded modalities
        """
        return list(self._encoders.keys())
    
    async def preload_encoders(self, modalities: Optional[List[Modality]] = None) -> None:
        """Preload encoders for specified modalities.
        
        Args:
            modalities: List of modalities to preload (default: all registered)
        """
        if modalities is None:
            modalities = self.get_registered_modalities()
        
        for modality in modalities:
            try:
                await self.get_encoder(modality)
            except Exception as e:
                logger.error(f"Failed to preload encoder for {modality.value}: {e}")
    
    def unload_encoder(self, modality: Modality) -> None:
        """Unload an encoder to free memory.
        
        Args:
            modality: Modality type
        """
        if modality in self._encoders:
            del self._encoders[modality]
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded encoder for {modality.value}")
    
    def unload_all_encoders(self) -> None:
        """Unload all encoders to free memory."""
        for modality in list(self._encoders.keys()):
            self.unload_encoder(modality)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory usage stats
        """
        stats = {
            "loaded_encoders": len(self._encoders),
            "registered_encoders": len(self._encoder_configs),
            "modalities": {
                modality.value: encoder.get_model_info()
                for modality, encoder in self._encoders.items()
            }
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        
        return stats


# Global embedding manager instance
embedding_manager = EmbeddingManager()
