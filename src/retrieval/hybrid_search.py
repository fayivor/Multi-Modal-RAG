"""Hybrid search combining vector similarity and BM25."""

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ..core.config import settings
from ..core.constants import Modality
from ..core.exceptions import SearchError
from .vector_store import SearchFilter, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index for keyword-based search."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        """Initialize BM25 index.
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.tokenized_corpus = []
    
    def build_index(self, documents: List[SearchResult]) -> None:
        """Build BM25 index from documents.
        
        Args:
            documents: List of search results to index
        """
        self.documents = documents
        self.document_ids = [doc.id for doc in documents]
        
        # Tokenize documents
        self.tokenized_corpus = []
        for doc in documents:
            # Combine content and relevant metadata for indexing
            text_parts = [doc.document.content]
            
            # Add metadata text if available
            metadata = doc.document.metadata
            if 'title' in metadata:
                text_parts.append(metadata['title'])
            if 'description' in metadata:
                text_parts.append(metadata['description'])
            
            combined_text = " ".join(text_parts)
            tokens = self._tokenize(combined_text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Built BM25 index with {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                doc_id = self.document_ids[idx]
                score = float(scores[idx])
                results.append((doc_id, score))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be improved with proper NLP tokenization
        import re
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens


class HybridSearcher:
    """Hybrid searcher combining vector similarity and BM25."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        bm25_top_k: int = 100,
        enable_reranking: bool = True,
    ) -> None:
        """Initialize hybrid searcher.
        
        Args:
            vector_store: Vector store for similarity search
            vector_weight: Weight for vector similarity scores
            bm25_weight: Weight for BM25 scores
            bm25_top_k: Number of BM25 results to consider
            enable_reranking: Whether to enable reranking
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_top_k = bm25_top_k
        self.enable_reranking = enable_reranking
        
        # BM25 indices for different modalities
        self.bm25_indices: Dict[Modality, BM25Index] = {}
        self._index_built = False
    
    async def build_bm25_indices(
        self,
        modalities: Optional[List[Modality]] = None,
        max_docs_per_modality: int = 10000,
    ) -> None:
        """Build BM25 indices for specified modalities.
        
        Args:
            modalities: List of modalities to build indices for
            max_docs_per_modality: Maximum documents per modality
        """
        if modalities is None:
            modalities = [Modality.TEXT, Modality.CODE]  # BM25 works best with text
        
        try:
            for modality in modalities:
                logger.info(f"Building BM25 index for {modality.value}")
                
                # Get documents for this modality
                filter_obj = SearchFilter(modality=modality)
                
                # Use a dummy vector to get all documents of this modality
                dummy_vector = np.zeros(512)  # Adjust dimension as needed
                
                documents = await self.vector_store.search(
                    query_vector=dummy_vector,
                    top_k=max_docs_per_modality,
                    filters=filter_obj,
                )
                
                if documents:
                    # Build BM25 index
                    bm25_index = BM25Index()
                    bm25_index.build_index(documents)
                    self.bm25_indices[modality] = bm25_index
                    
                    logger.info(f"Built BM25 index for {modality.value} with {len(documents)} documents")
                else:
                    logger.warning(f"No documents found for modality {modality.value}")
            
            self._index_built = True
            
        except Exception as e:
            raise SearchError(f"Failed to build BM25 indices: {e}")
    
    async def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Perform hybrid search.
        
        Args:
            query: Text query for BM25 search
            query_vector: Vector for similarity search
            top_k: Number of final results to return
            filters: Search filters
            **kwargs: Additional search parameters
            
        Returns:
            List of search results with hybrid scores
        """
        try:
            # Determine search modalities
            search_modalities = []
            if filters and filters.modality:
                search_modalities = [filters.modality]
            else:
                search_modalities = list(self.bm25_indices.keys())
                search_modalities.append(Modality.IMAGE)  # Always include image search
            
            # Perform vector search
            vector_results = await self._vector_search(
                query_vector, top_k * 2, filters, **kwargs
            )
            
            # Perform BM25 search for text-based modalities
            bm25_results = {}
            if self._index_built:
                for modality in search_modalities:
                    if modality in self.bm25_indices:
                        bm25_results[modality] = self.bm25_indices[modality].search(
                            query, self.bm25_top_k
                        )
            
            # Combine and score results
            combined_results = self._combine_results(
                vector_results, bm25_results, top_k
            )
            
            # Apply diversity filtering if enabled
            if len(combined_results) > top_k:
                combined_results = self._apply_diversity_filtering(
                    combined_results, top_k
                )
            
            return combined_results[:top_k]
            
        except Exception as e:
            raise SearchError(f"Hybrid search failed: {e}")
    
    async def _vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[SearchFilter],
        **kwargs
    ) -> List[SearchResult]:
        """Perform vector similarity search.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Search filters
            **kwargs: Additional parameters
            
        Returns:
            Vector search results
        """
        return await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            **kwargs
        )
    
    def _combine_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: Dict[Modality, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[SearchResult]:
        """Combine vector and BM25 results.
        
        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results by modality
            top_k: Number of final results
            
        Returns:
            Combined and scored results
        """
        # Create lookup for vector results
        vector_lookup = {result.id: result for result in vector_results}
        
        # Create lookup for BM25 results
        bm25_lookup = {}
        for modality, results in bm25_results.items():
            for doc_id, score in results:
                bm25_lookup[doc_id] = score
        
        # Normalize scores
        vector_scores = [result.score for result in vector_results]
        bm25_scores = list(bm25_lookup.values())
        
        vector_max = max(vector_scores) if vector_scores else 1.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        
        # Combine results
        combined_scores = {}
        
        # Add vector results
        for result in vector_results:
            normalized_vector_score = result.score / vector_max
            combined_scores[result.id] = {
                'result': result,
                'vector_score': normalized_vector_score,
                'bm25_score': 0.0,
                'hybrid_score': self.vector_weight * normalized_vector_score,
            }
        
        # Add BM25 scores
        for doc_id, bm25_score in bm25_lookup.items():
            normalized_bm25_score = bm25_score / bm25_max
            
            if doc_id in combined_scores:
                # Update existing result
                combined_scores[doc_id]['bm25_score'] = normalized_bm25_score
                combined_scores[doc_id]['hybrid_score'] = (
                    self.vector_weight * combined_scores[doc_id]['vector_score'] +
                    self.bm25_weight * normalized_bm25_score
                )
            else:
                # BM25-only result (shouldn't happen with current implementation)
                logger.warning(f"BM25-only result found: {doc_id}")
        
        # Sort by hybrid score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        # Update result scores and return
        final_results = []
        for item in sorted_results:
            result = item['result']
            result.score = item['hybrid_score']
            result.metadata.update({
                'vector_score': item['vector_score'],
                'bm25_score': item['bm25_score'],
                'hybrid_score': item['hybrid_score'],
            })
            final_results.append(result)
        
        return final_results
    
    def _apply_diversity_filtering(
        self,
        results: List[SearchResult],
        target_count: int,
        similarity_threshold: float = 0.8,
    ) -> List[SearchResult]:
        """Apply diversity filtering to reduce redundant results.
        
        Args:
            results: Input results
            target_count: Target number of results
            similarity_threshold: Similarity threshold for diversity
            
        Returns:
            Filtered results with diversity
        """
        if len(results) <= target_count:
            return results
        
        diverse_results = []
        used_content_hashes = set()
        
        for result in results:
            # Simple content-based diversity check
            content_hash = hash(result.document.content[:200])  # Use first 200 chars
            
            if content_hash not in used_content_hashes:
                diverse_results.append(result)
                used_content_hashes.add(content_hash)
                
                if len(diverse_results) >= target_count:
                    break
        
        # If we don't have enough diverse results, fill with remaining results
        if len(diverse_results) < target_count:
            for result in results:
                if result not in diverse_results:
                    diverse_results.append(result)
                    if len(diverse_results) >= target_count:
                        break
        
        return diverse_results
    
    def get_search_stats(self) -> Dict[str, any]:
        """Get search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "bm25_top_k": self.bm25_top_k,
            "enable_reranking": self.enable_reranking,
            "bm25_indices_built": self._index_built,
            "indexed_modalities": list(self.bm25_indices.keys()),
            "index_sizes": {
                modality.value: len(index.documents)
                for modality, index in self.bm25_indices.items()
            },
        }
