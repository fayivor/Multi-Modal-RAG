"""Image embedding encoder using CLIP and similar models."""

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..core.constants import Modality
from ..core.exceptions import EncodingError, ModelLoadError
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class ImageEncoder(BaseEncoder):
    """Image embedding encoder using CLIP model."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 8,  # Smaller batch size for images
        normalize: bool = True,
        image_size: int = 224,
    ) -> None:
        """Initialize the image encoder.
        
        Args:
            model_name: Name of the CLIP model
            device: Device to run the model on
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            image_size: Target image size for processing
        """
        super().__init__(model_name, device, batch_size, normalize)
        self.image_size = image_size
        self._processor = None
    
    async def load_model(self) -> None:
        """Load the CLIP model and processor."""
        try:
            logger.info(f"Loading image encoder: {self.model_name}")
            
            # Load model and processor in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            self._model, self._processor = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            # Move model to device
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            
            # Set model to evaluation mode
            self._model.eval()
            
            self._is_loaded = True
            logger.info(f"Successfully loaded image encoder: {self.model_name}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load image encoder {self.model_name}: {e}")
    
    def _load_model_sync(self) -> tuple:
        """Synchronously load the model and processor."""
        processor = CLIPProcessor.from_pretrained(self.model_name)
        model = CLIPModel.from_pretrained(self.model_name)
        return model, processor
    
    async def encode(
        self,
        inputs: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        **kwargs
    ) -> np.ndarray:
        """Encode image inputs into embeddings.
        
        Args:
            inputs: Image path(s), PIL Image(s), or mixed list
            **kwargs: Additional encoding parameters
            
        Returns:
            Numpy array of embeddings
        """
        await self.ensure_loaded()
        
        # Normalize inputs to list
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if not inputs:
            return np.array([])
        
        try:
            # Load and preprocess images
            images = []
            for inp in inputs:
                image = self._load_image(inp)
                images.append(image)
            
            # Process in batches
            if len(images) <= self.batch_size:
                embeddings = await self._encode_batch(images, **kwargs)
            else:
                all_embeddings = []
                batches = self._batch_inputs(images)
                
                for batch in batches:
                    batch_embeddings = await self._encode_batch(batch, **kwargs)
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode image inputs: {e}")
    
    def _encode_batch_sync(self, batch: List[Image.Image], kwargs: Dict) -> np.ndarray:
        """Synchronous batch encoding method.
        
        Args:
            batch: Batch of PIL Images to encode
            kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        # Process images
        inputs = self._processor(
            images=batch,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image embeddings
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()
    
    def _load_image(self, input_image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load and preprocess an image.
        
        Args:
            input_image: Image path or PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image)
        elif isinstance(input_image, Image.Image):
            image = input_image
        else:
            raise ValueError(f"Unsupported image input type: {type(input_image)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return default dimensions for common CLIP models
            model_dims = {
                "openai/clip-vit-base-patch32": 512,
                "openai/clip-vit-base-patch16": 512,
                "openai/clip-vit-large-patch14": 768,
                "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
                "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": 768,
            }
            return model_dims.get(self.model_name, 512)
        
        return self._model.config.projection_dim
    
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles.
        
        Returns:
            Image modality
        """
        return Modality.IMAGE
    
    def encode_with_text(
        self,
        images: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Encode images and texts together for multimodal similarity.
        
        Args:
            images: Image inputs
            texts: Text descriptions
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary with image and text embeddings
        """
        # Encode images
        image_embeddings = self.encode(images, **kwargs)
        
        # Encode texts using CLIP text encoder
        text_embeddings = self.encode_text(texts, **kwargs)
        
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
    
    async def encode_text(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """Encode text using CLIP text encoder.
        
        Args:
            texts: Text string or list of text strings
            **kwargs: Additional encoding parameters
            
        Returns:
            Text embeddings
        """
        await self.ensure_loaded()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            # Process in batches
            if len(texts) <= self.batch_size:
                embeddings = await self._encode_text_batch(texts, **kwargs)
            else:
                all_embeddings = []
                batches = self._batch_inputs(texts)
                
                for batch in batches:
                    batch_embeddings = await self._encode_text_batch(batch, **kwargs)
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode text inputs: {e}")
    
    async def _encode_text_batch(self, batch: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts using CLIP text encoder.
        
        Args:
            batch: Batch of text strings
            **kwargs: Additional encoding parameters
            
        Returns:
            Text embeddings
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._encode_text_batch_sync, batch, kwargs
        )
    
    def _encode_text_batch_sync(self, batch: List[str], kwargs: Dict) -> np.ndarray:
        """Synchronous text batch encoding method.
        
        Args:
            batch: Batch of text strings
            kwargs: Additional encoding parameters
            
        Returns:
            Text embeddings
        """
        # Process texts
        inputs = self._processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text embeddings
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Text embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings if not already normalized
        if not self.normalize:
            image_embeddings = image_embeddings / np.linalg.norm(
                image_embeddings, axis=1, keepdims=True
            )
            text_embeddings = text_embeddings / np.linalg.norm(
                text_embeddings, axis=1, keepdims=True
            )
        
        # Compute cosine similarity
        similarity = np.dot(image_embeddings, text_embeddings.T)
        return similarity


class DiagramEncoder(ImageEncoder):
    """Specialized encoder for technical diagrams and charts."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize the diagram encoder."""
        super().__init__(**kwargs)
    
    async def encode(
        self,
        inputs: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        enhance_diagrams: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode diagram images with optional enhancement.
        
        Args:
            inputs: Image inputs
            enhance_diagrams: Whether to enhance diagrams for better encoding
            **kwargs: Additional encoding parameters
            
        Returns:
            Diagram embeddings
        """
        if enhance_diagrams:
            # Preprocess images for better diagram understanding
            if not isinstance(inputs, list):
                inputs = [inputs]
            
            enhanced_inputs = []
            for inp in inputs:
                image = self._load_image(inp)
                enhanced_image = self._enhance_diagram(image)
                enhanced_inputs.append(enhanced_image)
            
            inputs = enhanced_inputs
        
        return await super().encode(inputs, **kwargs)
    
    def _enhance_diagram(self, image: Image.Image) -> Image.Image:
        """Enhance diagram image for better encoding.
        
        Args:
            image: Input diagram image
            
        Returns:
            Enhanced image
        """
        # Simple enhancement - in practice, you might use more sophisticated methods
        import cv2
        import numpy as np
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to enhance text and lines
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(enhanced)
