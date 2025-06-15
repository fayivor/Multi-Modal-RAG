"""Integration tests for the multi-modal RAG system."""

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings
from src.embeddings.multi_modal import multi_modal_system
from src.retrieval.vector_store import vector_store


class TestMultiModalRAGIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture(scope="class", autouse=True)
    async def setup_system(self):
        """Set up the system for testing."""
        # Initialize vector store collection
        try:
            await vector_store.create_collection(
                dimension=384,  # Default dimension for MiniLM
                distance_metric="cosine"
            )
        except Exception:
            pass  # Collection might already exist
        
        yield
        
        # Cleanup after tests
        # In a real test, you might want to clean up test data
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
    
    def test_search_without_documents(self, client):
        """Test search when no documents are uploaded."""
        response = client.post("/api/v1/search", json={
            "query": "test query",
            "modalities": ["text"],
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["results"] == []
        assert data["total_results"] == 0
    
    def test_document_upload_and_search(self, client):
        """Test document upload and subsequent search."""
        # Create a test document
        test_content = """
        # FastAPI Authentication Guide
        
        This guide explains how to implement authentication in FastAPI applications.
        
        ## Basic Authentication
        
        ```python
        from fastapi import FastAPI, Depends, HTTPException
        from fastapi.security import HTTPBearer
        
        app = FastAPI()
        security = HTTPBearer()
        
        @app.get("/protected")
        async def protected_route(token: str = Depends(security)):
            if not validate_token(token):
                raise HTTPException(status_code=401, detail="Invalid token")
            return {"message": "Access granted"}
        ```
        
        ## JWT Authentication
        
        For more secure authentication, use JWT tokens:
        
        ```python
        import jwt
        from datetime import datetime, timedelta
        
        def create_jwt_token(user_id: str) -> str:
            payload = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(hours=24)
            }
            return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        ```
        """
        
        # Upload document
        files = {"file": ("test_auth_guide.md", test_content, "text/markdown")}
        data = {"metadata": '{"category": "authentication", "tags": ["fastapi", "security"]}'}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        # Check upload was successful
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert upload_data["status"] == "completed"
        assert upload_data["chunks_created"] > 0
        
        # Wait a moment for processing
        import time
        time.sleep(2)
        
        # Test search for uploaded content
        search_response = client.post("/api/v1/search", json={
            "query": "How to implement JWT authentication in FastAPI?",
            "modalities": ["text", "code"],
            "top_k": 5,
            "enable_reranking": True
        })
        
        assert search_response.status_code == 200
        search_data = search_response.json()
        
        # Should find relevant results
        assert search_data["total_results"] > 0
        assert len(search_data["results"]) > 0
        
        # Check that results contain relevant content
        found_jwt = False
        found_fastapi = False
        
        for result in search_data["results"]:
            content = result["document"]["content"].lower()
            if "jwt" in content:
                found_jwt = True
            if "fastapi" in content:
                found_fastapi = True
        
        assert found_jwt or found_fastapi, "Search results should contain relevant content"
    
    def test_chat_functionality(self, client):
        """Test chat functionality with context."""
        response = client.post("/api/v1/chat", json={
            "message": "What is JWT authentication and how do I implement it?",
            "stream_response": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert len(data["message"]) > 0
        assert "conversation_id" in data
    
    def test_query_analysis(self, client):
        """Test query analysis functionality."""
        response = client.post("/api/v1/search", json={
            "query": "def authenticate_user(username, password):",
            "modalities": ["code"],
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check query analysis
        if "query_analysis" in data and data["query_analysis"]:
            analysis = data["query_analysis"]
            assert "intent" in analysis
            assert "modalities" in analysis
            # Should detect code intent
            assert "code" in [m.lower() for m in analysis["modalities"]]
    
    def test_multi_modal_search(self, client):
        """Test search across multiple modalities."""
        response = client.post("/api/v1/search", json={
            "query": "authentication security implementation",
            "modalities": ["text", "code"],
            "top_k": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle multi-modal search without errors
        assert "results" in data
        assert "search_time_ms" in data
        assert data["search_time_ms"] > 0
    
    def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        # Test invalid modality
        response = client.post("/api/v1/search", json={
            "query": "test",
            "modalities": ["invalid_modality"],
            "top_k": 5
        })
        
        assert response.status_code == 422  # Validation error
        
        # Test empty query
        response = client.post("/api/v1/search", json={
            "query": "",
            "modalities": ["text"],
            "top_k": 5
        })
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid top_k
        response = client.post("/api/v1/search", json={
            "query": "test",
            "modalities": ["text"],
            "top_k": 0
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_system_info(self, client):
        """Test system information endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "application" in data
        assert "embeddings" in data
        assert "vector_store" in data
        assert "llm" in data
        assert "features" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint if enabled."""
        response = client.get("/metrics")
        
        # Metrics might be disabled in test environment
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            # Should return Prometheus format
            assert "text/plain" in response.headers.get("content-type", "")


@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async components directly."""
    
    async def test_embedding_system(self):
        """Test embedding system functionality."""
        # Test text embedding
        text_embedding = await multi_modal_system.encode(
            "This is a test sentence for embedding.",
            modality="text"
        )
        
        assert text_embedding is not None
        assert len(text_embedding.shape) == 1  # Should be 1D array
        assert text_embedding.shape[0] > 0  # Should have dimensions
    
    async def test_vector_store_operations(self):
        """Test vector store operations."""
        # Test health check
        is_healthy = await vector_store.health_check()
        assert isinstance(is_healthy, bool)
        
        # Test collection info
        try:
            info = await vector_store.get_collection_info()
            assert isinstance(info, dict)
        except Exception:
            # Collection might not exist in test environment
            pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
