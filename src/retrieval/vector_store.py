"""Vector store interface and implementations."""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from ..core.config import settings
from ..core.constants import Modality, VectorStoreType
from ..core.exceptions import VectorStoreError
from ..ingestion.base import Document

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Search result from vector store."""
    
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    document: Document = Field(..., description="Retrieved document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchFilter(BaseModel):
    """Filter for vector search."""
    
    modality: Optional[Modality] = None
    source: Optional[str] = None
    date_range: Optional[Dict[str, Any]] = None
    custom_filters: Dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, collection_name: str = "documents") -> None:
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection/index
        """
        self.collection_name = collection_name
    
    @abstractmethod
    async def create_collection(
        self,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new collection/index.
        
        Args:
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)
            **kwargs: Additional collection parameters
        """
        pass
    
    @abstractmethod
    async def upsert(self, documents: List[Document]) -> None:
        """Insert or update documents in the vector store.
        
        Args:
            documents: List of documents to upsert
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Search filters
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents by IDs.
        
        Args:
            document_ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Collection information
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = None,
        port: int = None,
        api_key: str = None,
        timeout: int = None,
    ) -> None:
        """Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            host: Qdrant host
            port: Qdrant port
            api_key: API key for authentication
            timeout: Request timeout
        """
        super().__init__(collection_name)
        
        self.host = host or settings.vector_store.qdrant_host
        self.port = port or settings.vector_store.qdrant_port
        self.api_key = api_key or settings.vector_store.qdrant_api_key
        self.timeout = timeout or settings.vector_store.qdrant_timeout
        
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            timeout=self.timeout,
        )
    
    async def create_collection(
        self,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new Qdrant collection.
        
        Args:
            dimension: Vector dimension
            distance_metric: Distance metric
            **kwargs: Additional collection parameters
        """
        try:
            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            
            distance = distance_map.get(distance_metric, Distance.COSINE)
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance,
                ),
                **kwargs
            )
            
            logger.info(f"Created Qdrant collection: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create Qdrant collection: {e}")
    
    async def upsert(self, documents: List[Document]) -> None:
        """Upsert documents into Qdrant.
        
        Args:
            documents: List of documents to upsert
        """
        if not documents:
            return
        
        try:
            points = []
            for doc in documents:
                if doc.embeddings is None:
                    logger.warning(f"Document {doc.chunk_id} has no embeddings, skipping")
                    continue
                
                # Generate ID if not present
                doc_id = doc.chunk_id or str(uuid.uuid4())
                
                # Prepare metadata
                payload = {
                    "content": doc.content,
                    "modality": doc.modality.value,
                    "metadata": doc.metadata,
                }
                
                # Create point
                point = models.PointStruct(
                    id=doc_id,
                    vector=doc.embeddings,
                    payload=payload,
                )
                points.append(point)
            
            if points:
                # Upsert points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                
                logger.info(f"Upserted {len(points)} documents to Qdrant")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert documents to Qdrant: {e}")
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search in Qdrant collection.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Search filters
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        try:
            # Prepare query vector
            if query_vector.ndim > 1:
                query_vector = query_vector.flatten()
            
            # Prepare filters
            qdrant_filter = None
            if filters:
                conditions = []
                
                if filters.modality:
                    conditions.append(
                        models.FieldCondition(
                            key="modality",
                            match=models.MatchValue(value=filters.modality.value)
                        )
                    )
                
                if filters.source:
                    conditions.append(
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filters.source)
                        )
                    )
                
                # Add custom filters
                for key, value in filters.custom_filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if conditions:
                    qdrant_filter = models.Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=qdrant_filter,
                limit=top_k,
                **kwargs
            )
            
            # Convert results
            results = []
            for hit in search_result:
                # Reconstruct document
                payload = hit.payload
                document = Document(
                    content=payload["content"],
                    metadata=payload["metadata"],
                    modality=Modality(payload["modality"]),
                    chunk_id=str(hit.id),
                )
                
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    document=document,
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search in Qdrant: {e}")
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from Qdrant.
        
        Args:
            document_ids: List of document IDs to delete
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=document_ids,
                ),
            )
            
            logger.info(f"Deleted {len(document_ids)} documents from Qdrant")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents from Qdrant: {e}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get Qdrant collection information.
        
        Returns:
            Collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": info.config.params.dict() if info.config.params else {},
                    "hnsw_config": info.config.hnsw_config.dict() if info.config.hnsw_config else {},
                    "optimizer_config": info.config.optimizer_config.dict() if info.config.optimizer_config else {},
                },
            }
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get Qdrant collection info: {e}")
    
    async def health_check(self) -> bool:
        """Check Qdrant health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create_vector_store(
        store_type: VectorStoreType = None,
        collection_name: str = "documents",
        **kwargs
    ) -> VectorStore:
        """Create a vector store instance.
        
        Args:
            store_type: Type of vector store to create
            collection_name: Name of the collection
            **kwargs: Additional parameters for the vector store
            
        Returns:
            Vector store instance
        """
        if store_type is None:
            store_type = settings.vector_store.type
        
        if store_type == VectorStoreType.QDRANT:
            return QdrantVectorStore(collection_name=collection_name, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


# Global vector store instance
vector_store = VectorStoreFactory.create_vector_store()
