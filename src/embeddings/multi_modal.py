"""Multi-modal embedding system with unified embedding space."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.config import settings
from ..core.constants import Modality
from ..core.exceptions import EncodingError, ModelLoadError
from .base import BaseEncoder, EmbeddingManager, embedding_manager
from .code_encoder import CodeEncoder
from .image_encoder import ImageEncoder
from .table_encoder import SimpleTableEncoder, TableEncoder
from .text_encoder import TextEncoder

logger = logging.getLogger(__name__)


class MultiModalEmbeddingSystem:
    """Unified multi-modal embedding system."""
    
    def __init__(
        self,
        target_dimension: int = 512,
        alignment_method: str = "pca",
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the multi-modal embedding system.
        
        Args:
            target_dimension: Target dimension for unified embeddings
            alignment_method: Method for aligning embeddings (pca, linear)
            normalize_embeddings: Whether to normalize final embeddings
        """
        self.target_dimension = target_dimension
        self.alignment_method = alignment_method
        self.normalize_embeddings = normalize_embeddings
        
        # Alignment models for each modality
        self._alignment_models: Dict[Modality, Any] = {}
        self._scalers: Dict[Modality, StandardScaler] = {}
        self._is_fitted = False
        
        # Initialize embedding manager
        self.embedding_manager = embedding_manager
        self._setup_encoders()
    
    def _setup_encoders(self) -> None:
        """Set up encoders for each modality."""
        # Register text encoder
        self.embedding_manager.register_encoder(
            Modality.TEXT,
            TextEncoder,
            model_name=settings.embeddings.text_model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
        
        # Register code encoder
        self.embedding_manager.register_encoder(
            Modality.CODE,
            CodeEncoder,
            model_name=settings.embeddings.code_model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
        
        # Register image encoder
        self.embedding_manager.register_encoder(
            Modality.IMAGE,
            ImageEncoder,
            model_name=settings.embeddings.image_model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
        
        # Register table encoder (using simple text-based approach)
        text_encoder = TextEncoder(
            model_name=settings.embeddings.text_model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
        
        self.embedding_manager.register_encoder(
            Modality.TABLE,
            SimpleTableEncoder,
            text_encoder=text_encoder,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
    
    async def encode(
        self,
        inputs: Union[str, List[str], Any],
        modality: Modality,
        align_to_unified_space: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode inputs using the appropriate encoder.
        
        Args:
            inputs: Input data to encode
            modality: Modality type
            align_to_unified_space: Whether to align to unified embedding space
            **kwargs: Additional encoding parameters
            
        Returns:
            Embeddings (aligned to unified space if requested)
        """
        try:
            # Get raw embeddings
            embeddings = await self.embedding_manager.encode(
                inputs, modality, **kwargs
            )
            
            # Align to unified space if requested and fitted
            if align_to_unified_space and self._is_fitted:
                embeddings = self._align_embeddings(embeddings, modality)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode {modality.value} inputs: {e}")
    
    async def encode_batch_multimodal(
        self,
        batch: List[Dict[str, Any]],
        align_to_unified_space: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Encode a batch of multi-modal inputs.
        
        Args:
            batch: List of dicts with 'content', 'modality', and optional kwargs
            align_to_unified_space: Whether to align to unified space
            
        Returns:
            Dictionary with embeddings for each modality
        """
        # Group inputs by modality
        modality_groups = {}
        for item in batch:
            modality = item['modality']
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(item)
        
        # Encode each modality group
        results = {}
        for modality, items in modality_groups.items():
            contents = [item['content'] for item in items]
            kwargs = {}
            
            # Merge kwargs from all items (assuming they're consistent)
            for item in items:
                kwargs.update(item.get('kwargs', {}))
            
            embeddings = await self.encode(
                contents, modality, align_to_unified_space, **kwargs
            )
            results[modality.value] = embeddings
        
        return results
    
    def fit_alignment(
        self,
        training_data: Dict[Modality, np.ndarray],
        method: str = None,
    ) -> None:
        """Fit alignment models to create unified embedding space.
        
        Args:
            training_data: Dictionary mapping modalities to embedding samples
            method: Alignment method to use (overrides default)
        """
        if method is None:
            method = self.alignment_method
        
        logger.info(f"Fitting alignment models using {method} method")
        
        try:
            for modality, embeddings in training_data.items():
                if len(embeddings) == 0:
                    continue
                
                # Fit scaler
                scaler = StandardScaler()
                scaled_embeddings = scaler.fit_transform(embeddings)
                self._scalers[modality] = scaler
                
                # Fit alignment model
                if method == "pca":
                    alignment_model = PCA(
                        n_components=min(self.target_dimension, embeddings.shape[1])
                    )
                    alignment_model.fit(scaled_embeddings)
                    self._alignment_models[modality] = alignment_model
                
                elif method == "linear":
                    # Simple linear projection (identity for now)
                    # In practice, you might train a linear layer to align embeddings
                    from sklearn.random_projection import GaussianRandomProjection
                    alignment_model = GaussianRandomProjection(
                        n_components=self.target_dimension
                    )
                    alignment_model.fit(scaled_embeddings)
                    self._alignment_models[modality] = alignment_model
                
                else:
                    raise ValueError(f"Unsupported alignment method: {method}")
                
                logger.info(
                    f"Fitted alignment for {modality.value}: "
                    f"{embeddings.shape[1]} -> {self.target_dimension} dimensions"
                )
            
            self._is_fitted = True
            logger.info("Successfully fitted all alignment models")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to fit alignment models: {e}")
    
    def _align_embeddings(
        self,
        embeddings: np.ndarray,
        modality: Modality
    ) -> np.ndarray:
        """Align embeddings to unified space.
        
        Args:
            embeddings: Raw embeddings
            modality: Modality type
            
        Returns:
            Aligned embeddings
        """
        if modality not in self._alignment_models:
            logger.warning(f"No alignment model for {modality.value}, returning raw embeddings")
            return embeddings
        
        try:
            # Scale embeddings
            if modality in self._scalers:
                scaled_embeddings = self._scalers[modality].transform(embeddings)
            else:
                scaled_embeddings = embeddings
            
            # Apply alignment transformation
            aligned_embeddings = self._alignment_models[modality].transform(scaled_embeddings)
            
            # Pad or truncate to target dimension
            if aligned_embeddings.shape[1] < self.target_dimension:
                # Pad with zeros
                padding = np.zeros((aligned_embeddings.shape[0], 
                                  self.target_dimension - aligned_embeddings.shape[1]))
                aligned_embeddings = np.hstack([aligned_embeddings, padding])
            elif aligned_embeddings.shape[1] > self.target_dimension:
                # Truncate
                aligned_embeddings = aligned_embeddings[:, :self.target_dimension]
            
            # Normalize if requested
            if self.normalize_embeddings:
                norms = np.linalg.norm(aligned_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                aligned_embeddings = aligned_embeddings / norms
            
            return aligned_embeddings
            
        except Exception as e:
            logger.error(f"Failed to align {modality.value} embeddings: {e}")
            return embeddings
    
    def compute_cross_modal_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        modality1: Modality,
        modality2: Modality,
    ) -> np.ndarray:
        """Compute similarity between embeddings from different modalities.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            modality1: Modality of first embeddings
            modality2: Modality of second embeddings
            
        Returns:
            Cross-modal similarity matrix
        """
        # Align embeddings to unified space
        if self._is_fitted:
            aligned_emb1 = self._align_embeddings(embeddings1, modality1)
            aligned_emb2 = self._align_embeddings(embeddings2, modality2)
        else:
            logger.warning("Alignment models not fitted, using raw embeddings")
            aligned_emb1 = embeddings1
            aligned_emb2 = embeddings2
        
        # Compute cosine similarity
        # Normalize embeddings
        norm1 = np.linalg.norm(aligned_emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(aligned_emb2, axis=1, keepdims=True)
        
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)
        
        normalized_emb1 = aligned_emb1 / norm1
        normalized_emb2 = aligned_emb2 / norm2
        
        similarity = np.dot(normalized_emb1, normalized_emb2.T)
        return similarity
    
    async def preload_all_encoders(self) -> None:
        """Preload all encoders for faster inference."""
        await self.embedding_manager.preload_encoders()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the multi-modal system.
        
        Returns:
            System information dictionary
        """
        return {
            "target_dimension": self.target_dimension,
            "alignment_method": self.alignment_method,
            "normalize_embeddings": self.normalize_embeddings,
            "is_fitted": self._is_fitted,
            "registered_modalities": [m.value for m in self.embedding_manager.get_registered_modalities()],
            "loaded_modalities": [m.value for m in self.embedding_manager.get_loaded_modalities()],
            "alignment_models": list(self._alignment_models.keys()),
            "memory_usage": self.embedding_manager.get_memory_usage(),
        }
    
    def save_alignment_models(self, path: str) -> None:
        """Save alignment models to disk.
        
        Args:
            path: Path to save models
        """
        import pickle
        from pathlib import Path
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save alignment models
        with open(save_path / "alignment_models.pkl", "wb") as f:
            pickle.dump(self._alignment_models, f)
        
        # Save scalers
        with open(save_path / "scalers.pkl", "wb") as f:
            pickle.dump(self._scalers, f)
        
        # Save metadata
        metadata = {
            "target_dimension": self.target_dimension,
            "alignment_method": self.alignment_method,
            "normalize_embeddings": self.normalize_embeddings,
            "is_fitted": self._is_fitted,
        }
        
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved alignment models to {save_path}")
    
    def load_alignment_models(self, path: str) -> None:
        """Load alignment models from disk.
        
        Args:
            path: Path to load models from
        """
        import pickle
        from pathlib import Path
        
        load_path = Path(path)
        
        # Load alignment models
        with open(load_path / "alignment_models.pkl", "rb") as f:
            self._alignment_models = pickle.load(f)
        
        # Load scalers
        with open(load_path / "scalers.pkl", "rb") as f:
            self._scalers = pickle.load(f)
        
        # Load metadata
        with open(load_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.target_dimension = metadata["target_dimension"]
            self.alignment_method = metadata["alignment_method"]
            self.normalize_embeddings = metadata["normalize_embeddings"]
            self._is_fitted = metadata["is_fitted"]
        
        logger.info(f"Loaded alignment models from {load_path}")


# Global multi-modal embedding system
multi_modal_system = MultiModalEmbeddingSystem()
