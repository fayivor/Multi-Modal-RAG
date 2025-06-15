"""Image processor for OCR and diagram extraction."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ..core.constants import Modality, OCR_DPI, OCR_LANGUAGES
from ..core.exceptions import ImageProcessingError, UnsupportedFileTypeError
from .base import BaseProcessor, Document, ProcessingResult

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """Processor for extracting text from images using OCR."""
    
    def __init__(
        self,
        dpi: int = OCR_DPI,
        languages: List[str] = None,
        preprocess: bool = True,
    ) -> None:
        """Initialize the image processor.
        
        Args:
            dpi: DPI for OCR processing
            languages: List of languages for OCR (default: English)
            preprocess: Whether to preprocess images for better OCR
        """
        super().__init__()
        self.dpi = dpi
        self.languages = languages or OCR_LANGUAGES
        self.preprocess = preprocess
        self._supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    def supports_file(self, file_path: Path) -> bool:
        """Check if this processor supports the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is supported, False otherwise
        """
        return file_path.suffix.lower() in self._supported_formats
    
    async def process(self, file_path: Path) -> ProcessingResult:
        """Process an image file and extract text using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Processing result with extracted text documents
        """
        if not self.supports_file(file_path):
            return ProcessingResult(
                status="failed",
                error_message=f"Unsupported file type: {file_path.suffix}"
            )
        
        self._validate_file(file_path)
        
        try:
            # Run image processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None, self._process_image_sync, file_path
            )
            
            return ProcessingResult(
                documents=documents,
                status="completed",
                metadata={
                    "processor": self.__class__.__name__,
                    "ocr_languages": self.languages,
                    "dpi": self.dpi,
                    "preprocessed": self.preprocess,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return ProcessingResult(
                status="failed",
                error_message=str(e)
            )
    
    def _process_image_sync(self, file_path: Path) -> List[Document]:
        """Synchronous image processing method.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of documents with extracted text
        """
        try:
            # Load image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image if enabled
            if self.preprocess:
                image = self._preprocess_image(image)
            
            # Extract text using OCR
            ocr_config = f'--oem 3 --psm 6 -l {"+".join(self.languages)}'
            text = pytesseract.image_to_string(image, config=ocr_config)
            
            # Get detailed OCR data for confidence scores
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config=ocr_config
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Create metadata
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix.lower(),
                "content_type": f"image/{file_path.suffix[1:].lower()}",
                "image_width": image.width,
                "image_height": image.height,
                "image_mode": image.mode,
                "ocr_confidence": avg_confidence,
                "ocr_languages": self.languages,
                "character_count": len(text),
                "word_count": len(text.split()),
                "preprocessed": self.preprocess,
            }
            
            if not text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return []
            
            return [Document(
                content=text.strip(),
                metadata=metadata,
                modality=Modality.IMAGE
            )]
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to process image {file_path}: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(cleaned)
            
            # Resize image if it's too small (OCR works better on larger images)
            width, height = processed_image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                processed_image = processed_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original image if preprocessing fails
    
    def extract_text_regions(self, file_path: Path) -> List[Tuple[str, dict]]:
        """Extract text regions with bounding boxes from an image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of tuples containing (text, bounding_box_info)
        """
        try:
            image = Image.open(file_path)
            
            if self.preprocess:
                image = self._preprocess_image(image)
            
            # Get detailed OCR data with bounding boxes
            ocr_config = f'--oem 3 --psm 6 -l {"+".join(self.languages)}'
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config=ocr_config
            )
            
            regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Filter low confidence detections
                    text = data['text'][i].strip()
                    if text:
                        bbox_info = {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'confidence': data['conf'][i],
                            'level': data['level'][i],
                        }
                        regions.append((text, bbox_info))
            
            return regions
            
        except Exception as e:
            logger.error(f"Failed to extract text regions from {file_path}: {e}")
            return []


class DiagramProcessor(ImageProcessor):
    """Specialized processor for technical diagrams and charts."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize the diagram processor."""
        super().__init__(**kwargs)
    
    async def process(self, file_path: Path) -> ProcessingResult:
        """Process a diagram image with specialized handling.
        
        Args:
            file_path: Path to the diagram image
            
        Returns:
            Processing result with extracted information
        """
        # First, run standard OCR
        result = await super().process(file_path)
        
        if result.status == "completed" and result.documents:
            # Add diagram-specific metadata
            for doc in result.documents:
                doc.metadata.update({
                    "diagram_type": self._detect_diagram_type(file_path),
                    "is_diagram": True,
                })
        
        return result
    
    def _detect_diagram_type(self, file_path: Path) -> str:
        """Detect the type of diagram based on content analysis.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Detected diagram type
        """
        # This is a simplified implementation
        # In practice, you might use ML models to classify diagram types
        filename_lower = file_path.name.lower()
        
        if any(keyword in filename_lower for keyword in ['flowchart', 'flow', 'process']):
            return "flowchart"
        elif any(keyword in filename_lower for keyword in ['architecture', 'arch', 'system']):
            return "architecture"
        elif any(keyword in filename_lower for keyword in ['uml', 'class', 'sequence']):
            return "uml"
        elif any(keyword in filename_lower for keyword in ['network', 'topology']):
            return "network"
        else:
            return "unknown"


class ImageProcessorFactory:
    """Factory for creating image processors."""
    
    @classmethod
    def get_processor(cls, file_path: Path, **kwargs) -> Optional[ImageProcessor]:
        """Get the appropriate image processor for a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for processor initialization
            
        Returns:
            ImageProcessor instance or None if not supported
        """
        supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        
        if file_path.suffix.lower() in supported_formats:
            # Check if it's likely a diagram based on filename
            filename_lower = file_path.name.lower()
            diagram_keywords = ['diagram', 'chart', 'flowchart', 'architecture', 'uml']
            
            if any(keyword in filename_lower for keyword in diagram_keywords):
                return DiagramProcessor(**kwargs)
            else:
                return ImageProcessor(**kwargs)
        
        return None
    
    @classmethod
    def supports_file(cls, file_path: Path) -> bool:
        """Check if a file is supported by any image processor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return file_path.suffix.lower() in supported_formats
