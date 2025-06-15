"""Cross-encoder reranking for improving search results."""

import asyncio
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core.config import settings
from ..core.exceptions import RerankingError
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker:
    """Base class for reranking models."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        """Initialize the reranker.
        
        Args:
            model_name: Name of the reranking model
            device: Device to run the model on
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._is_loaded = False
    
    async def load_model(self) -> None:
        """Load the reranking model."""
        raise NotImplementedError
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        raise NotImplementedError
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run the model on
            batch_size: Batch size for reranking
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device, batch_size)
        self.max_length = max_length
    
    async def load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder reranker: {self.model_name}")
            
            # Load model in thread pool
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            self._is_loaded = True
            logger.info(f"Successfully loaded cross-encoder reranker: {self.model_name}")
            
        except Exception as e:
            raise RerankingError(f"Failed to load cross-encoder model {self.model_name}: {e}")
    
    def _load_model_sync(self) -> CrossEncoder:
        """Synchronously load the model."""
        return CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        if not self._is_loaded:
            await self.load_model()
        
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                # Combine content with relevant metadata
                doc_text = result.document.content
                
                # Add title if available
                if 'title' in result.document.metadata:
                    doc_text = f"{result.document.metadata['title']}\n{doc_text}"
                
                pairs.append([query, doc_text])
            
            # Get reranking scores
            if len(pairs) <= self.batch_size:
                scores = await self._score_batch(pairs)
            else:
                # Process in batches
                all_scores = []
                for i in range(0, len(pairs), self.batch_size):
                    batch = pairs[i:i + self.batch_size]
                    batch_scores = await self._score_batch(batch)
                    all_scores.extend(batch_scores)
                scores = all_scores
            
            # Update results with reranking scores
            for i, result in enumerate(results):
                result.metadata['rerank_score'] = float(scores[i])
                result.metadata['original_score'] = result.score
                result.score = float(scores[i])  # Update main score
            
            # Sort by reranking score
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            raise RerankingError(f"Failed to rerank results: {e}")
    
    async def _score_batch(self, pairs: List[List[str]]) -> List[float]:
        """Score a batch of query-document pairs.
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._score_batch_sync, pairs)
    
    def _score_batch_sync(self, pairs: List[List[str]]) -> List[float]:
        """Synchronously score a batch of pairs."""
        scores = self._model.predict(pairs)
        return scores.tolist() if isinstance(scores, np.ndarray) else scores


class TransformerReranker(BaseReranker):
    """Transformer-based reranker using HuggingFace models."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """Initialize the transformer reranker.
        
        Args:
            model_name: Name of the transformer model
            device: Device to run the model on
            batch_size: Batch size for reranking
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device, batch_size)
        self.max_length = max_length
        self._tokenizer = None
    
    async def load_model(self) -> None:
        """Load the transformer model."""
        try:
            logger.info(f"Loading transformer reranker: {self.model_name}")
            
            # Load model and tokenizer in thread pool
            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            # Move model to device
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            
            self._model.eval()
            self._is_loaded = True
            logger.info(f"Successfully loaded transformer reranker: {self.model_name}")
            
        except Exception as e:
            raise RerankingError(f"Failed to load transformer model {self.model_name}: {e}")
    
    def _load_model_sync(self) -> Tuple:
        """Synchronously load the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return model, tokenizer
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results using transformer model.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        if not self._is_loaded:
            await self.load_model()
        
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        try:
            # Prepare inputs
            texts = []
            for result in results:
                # Combine query and document
                doc_text = result.document.content
                combined_text = f"{query} [SEP] {doc_text}"
                texts.append(combined_text)
            
            # Get relevance scores
            scores = await self._score_texts(texts)
            
            # Update results with reranking scores
            for i, result in enumerate(results):
                result.metadata['rerank_score'] = float(scores[i])
                result.metadata['original_score'] = result.score
                result.score = float(scores[i])
            
            # Sort by reranking score
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            raise RerankingError(f"Failed to rerank results: {e}")
    
    async def _score_texts(self, texts: List[str]) -> List[float]:
        """Score a list of texts.
        
        Args:
            texts: List of texts to score
            
        Returns:
            List of relevance scores
        """
        if len(texts) <= self.batch_size:
            return await self._score_batch_texts(texts)
        else:
            # Process in batches
            all_scores = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_scores = await self._score_batch_texts(batch)
                all_scores.extend(batch_scores)
            return all_scores
    
    async def _score_batch_texts(self, texts: List[str]) -> List[float]:
        """Score a batch of texts.
        
        Args:
            texts: Batch of texts to score
            
        Returns:
            List of relevance scores
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._score_batch_texts_sync, texts)
    
    def _score_batch_texts_sync(self, texts: List[str]) -> List[float]:
        """Synchronously score a batch of texts."""
        # Tokenize
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Use positive class probability as relevance score
            scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        
        return scores.cpu().numpy().tolist()


class RerankingPipeline:
    """Pipeline for applying multiple reranking strategies."""
    
    def __init__(
        self,
        rerankers: List[BaseReranker],
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize the reranking pipeline.
        
        Args:
            rerankers: List of reranker instances
            weights: Weights for combining reranker scores
        """
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Apply multiple rerankers and combine scores.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        # Apply each reranker
        all_scores = []
        for reranker in self.rerankers:
            reranked_results = await reranker.rerank(query, results.copy())
            scores = [r.score for r in reranked_results]
            all_scores.append(scores)
        
        # Normalize scores
        normalized_scores = []
        for scores in all_scores:
            max_score = max(scores) if scores else 1.0
            normalized = [s / max_score for s in scores]
            normalized_scores.append(normalized)
        
        # Combine scores using weights
        combined_scores = []
        for i in range(len(results)):
            weighted_score = sum(
                weight * normalized_scores[j][i]
                for j, weight in enumerate(self.weights)
            )
            combined_scores.append(weighted_score)
        
        # Update results with combined scores
        for i, result in enumerate(results):
            result.metadata['combined_rerank_score'] = combined_scores[i]
            result.score = combined_scores[i]
        
        # Sort by combined score
        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        return reranked_results[:top_k]


# Factory function for creating rerankers
def create_reranker(
    reranker_type: str = "cross_encoder",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseReranker:
    """Create a reranker instance.
    
    Args:
        reranker_type: Type of reranker (cross_encoder, transformer)
        model_name: Model name (uses default if None)
        **kwargs: Additional arguments for the reranker
        
    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(
            model_name=model_name or default_model,
            **kwargs
        )
    elif reranker_type == "transformer":
        default_model = "microsoft/DialoGPT-medium"
        return TransformerReranker(
            model_name=model_name or default_model,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}")


# Global reranker instance
reranker = create_reranker(
    device=settings.embeddings.device,
    batch_size=settings.embeddings.batch_size,
)
