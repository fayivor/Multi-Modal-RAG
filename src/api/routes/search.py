"""Search API routes."""

import asyncio
import logging
import time
from typing import List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...core.config import settings
from ...core.exceptions import SearchError
from ...embeddings.multi_modal import multi_modal_system
from ...generation.citation_tracker import citation_tracker
from ...generation.llm_client import llm_client
from ...generation.prompt_builder import prompt_builder
from ...retrieval.hybrid_search import HybridSearcher
from ...retrieval.query_processor import query_processor
from ...retrieval.reranker import reranker
from ...retrieval.vector_store import SearchFilter, vector_store
from ..models.requests import ChatRequest, SearchRequest
from ..models.responses import (
    ChatResponse,
    DocumentInfo,
    GeneratedResponse,
    QueryAnalysisInfo,
    SearchResponse,
    SearchResultItem,
)

logger = structlog.get_logger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize hybrid searcher
hybrid_searcher = HybridSearcher(vector_store)


@router.post("/search", response_model=SearchResponse)
@limiter.limit("60/minute")
async def search_documents(
    request: SearchRequest,
    http_request: Request,
) -> SearchResponse:
    """Search for documents using multi-modal RAG.
    
    This endpoint performs hybrid search across multiple modalities including
    text, code, images, and tables. It uses vector similarity search combined
    with BM25 keyword search and optional reranking.
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    logger.info(
        "Search request received",
        query=request.query,
        modalities=[m.value for m in request.modalities],
        top_k=request.top_k,
        request_id=request_id,
    )
    
    try:
        # Analyze query
        query_analysis = query_processor.analyze_query(request.query)
        
        logger.info(
            "Query analysis completed",
            intent=query_analysis.intent.value,
            detected_modalities=[m.value for m in query_analysis.modalities],
            confidence=query_analysis.confidence,
            request_id=request_id,
        )
        
        # Use detected modalities if none specified
        search_modalities = request.modalities
        if not search_modalities or search_modalities == []:
            search_modalities = query_analysis.modalities
        
        # Generate embeddings for the query
        # Use the primary modality for embedding
        primary_modality = search_modalities[0] if search_modalities else query_analysis.modalities[0]
        
        query_embedding = await multi_modal_system.encode(
            request.query,
            primary_modality,
            align_to_unified_space=True
        )
        
        # Create search filters
        search_filter = SearchFilter()
        if len(search_modalities) == 1:
            search_filter.modality = search_modalities[0]
        
        # Add custom filters
        if request.filters:
            search_filter.custom_filters.update(request.filters)
        
        # Perform hybrid search
        search_results = await hybrid_searcher.search(
            query=request.query,
            query_vector=query_embedding,
            top_k=request.top_k * 2,  # Get more results for reranking
            filters=search_filter,
        )
        
        logger.info(
            "Initial search completed",
            results_count=len(search_results),
            request_id=request_id,
        )
        
        # Apply reranking if enabled
        reranked = False
        if request.enable_reranking and len(search_results) > 1:
            try:
                search_results = await reranker.rerank(
                    query=request.query,
                    results=search_results,
                    top_k=request.top_k,
                )
                reranked = True
                
                logger.info(
                    "Reranking completed",
                    final_results_count=len(search_results),
                    request_id=request_id,
                )
                
            except Exception as e:
                logger.warning(
                    "Reranking failed, using original results",
                    error=str(e),
                    request_id=request_id,
                )
        
        # Limit to requested number of results
        final_results = search_results[:request.top_k]
        
        # Convert to response format
        result_items = []
        for i, result in enumerate(final_results):
            doc_info = DocumentInfo(
                id=result.id,
                content=result.document.content,
                modality=result.document.modality,
                source=result.document.metadata.get('source', 'Unknown'),
                metadata=result.document.metadata,
                relevance_score=result.score,
            )
            
            result_item = SearchResultItem(
                document=doc_info,
                score=result.score,
                rank=i + 1,
                highlight=_create_highlight(result.document.content, request.query),
            )
            result_items.append(result_item)
        
        # Create citations if requested
        citations = []
        if request.include_citations and final_results:
            context_docs = prompt_builder.create_context_from_search_results(
                request.query, final_results
            ).documents
            citation_map = citation_tracker.create_citation_map(context_docs)
            citations = list(citation_map.citations.values())
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        # Create query analysis info
        query_analysis_info = QueryAnalysisInfo(
            intent=query_analysis.intent,
            modalities=query_analysis.modalities,
            keywords=query_analysis.keywords,
            entities=query_analysis.entities,
            confidence=query_analysis.confidence,
        )
        
        response = SearchResponse(
            query=request.query,
            results=result_items,
            total_results=len(search_results),
            query_analysis=query_analysis_info,
            search_time_ms=search_time_ms,
            reranked=reranked,
            citations=citations,
            metadata={
                "search_modalities": [m.value for m in search_modalities],
                "hybrid_search_used": True,
                "request_id": request_id,
            }
        )
        
        logger.info(
            "Search completed successfully",
            results_returned=len(final_results),
            search_time_ms=search_time_ms,
            request_id=request_id,
        )
        
        return response
        
    except SearchError as e:
        logger.error(
            "Search error occurred",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    except Exception as e:
        logger.error(
            "Unexpected error during search",
            error=str(e),
            request_id=request_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_with_documents(
    request: ChatRequest,
    http_request: Request,
) -> ChatResponse:
    """Chat with documents using conversational RAG.
    
    This endpoint provides a conversational interface that searches for relevant
    documents and generates responses using an LLM with proper citations.
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    logger.info(
        "Chat request received",
        message=request.message[:100] + "..." if len(request.message) > 100 else request.message,
        conversation_id=request.conversation_id,
        request_id=request_id,
    )
    
    try:
        # First, search for relevant documents
        search_request = SearchRequest(
            query=request.message,
            top_k=10,  # Get more documents for better context
            enable_reranking=True,
            include_citations=True,
        )
        
        # Perform search (reuse search logic)
        query_analysis = query_processor.analyze_query(request.message)
        primary_modality = query_analysis.modalities[0] if query_analysis.modalities else None
        
        if primary_modality:
            query_embedding = await multi_modal_system.encode(
                request.message,
                primary_modality,
                align_to_unified_space=True
            )
            
            search_results = await hybrid_searcher.search(
                query=request.message,
                query_vector=query_embedding,
                top_k=10,
            )
        else:
            search_results = []
        
        # Create context for LLM
        context = prompt_builder.create_context_from_search_results(
            request.message,
            search_results,
            query_analysis,
            max_documents=8,
        )
        
        # Select appropriate template based on intent
        template_name = prompt_builder.select_template_by_intent(query_analysis.intent)
        
        # Build prompt
        system_prompt, user_prompt = prompt_builder.build_prompt(
            context, template_name
        )
        
        # Generate response
        generation_start = time.time()
        
        if request.stream_response:
            # Return streaming response
            return StreamingResponse(
                _generate_streaming_chat_response(
                    system_prompt, user_prompt, context, request, request_id
                ),
                media_type="text/plain"
            )
        else:
            # Generate complete response
            llm_response = await llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
            )
            
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract citations
            citation_map = citation_tracker.create_citation_map(context.documents)
            cleaned_response, used_citations = citation_tracker.extract_citations_from_response(
                llm_response, citation_map
            )
            
            # Create generated response info
            generated_response = GeneratedResponse(
                answer=cleaned_response,
                citations=used_citations,
                confidence=0.8,  # Placeholder - could be calculated
                sources_used=len(used_citations),
                generation_time_ms=generation_time_ms,
            )
            
            # Convert search results to document info
            context_documents = [
                DocumentInfo(
                    id=result.id,
                    content=result.document.content,
                    modality=result.document.modality,
                    source=result.document.metadata.get('source', 'Unknown'),
                    metadata=result.document.metadata,
                    relevance_score=result.score,
                )
                for result in search_results[:5]  # Limit context docs in response
            ]
            
            response = ChatResponse(
                message=cleaned_response,
                conversation_id=request.conversation_id or f"conv_{request_id}",
                generated_response=generated_response,
                context_documents=context_documents,
                metadata={
                    "query_intent": query_analysis.intent.value,
                    "template_used": template_name,
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "request_id": request_id,
                }
            )
            
            logger.info(
                "Chat response generated successfully",
                response_length=len(cleaned_response),
                citations_count=len(used_citations),
                generation_time_ms=generation_time_ms,
                request_id=request_id,
            )
            
            return response
    
    except Exception as e:
        logger.error(
            "Error during chat processing",
            error=str(e),
            request_id=request_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Chat processing failed")


async def _generate_streaming_chat_response(
    system_prompt: str,
    user_prompt: str,
    context,
    request: ChatRequest,
    request_id: str,
):
    """Generate streaming chat response."""
    try:
        async for chunk in llm_client.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
        ):
            yield f"data: {chunk}\n\n"
        
        # Send final message with citations
        citation_map = citation_tracker.create_citation_map(context.documents)
        citations_text = citation_tracker.format_citations(
            list(citation_map.citations.values())
        )
        
        if citations_text:
            yield f"data: \n\n{citations_text}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(
            "Error in streaming response",
            error=str(e),
            request_id=request_id,
        )
        yield f"data: Error: {str(e)}\n\n"


def _create_highlight(content: str, query: str, max_length: int = 200) -> str:
    """Create highlighted snippet from content."""
    query_words = query.lower().split()
    content_lower = content.lower()
    
    # Find the best position to start the highlight
    best_pos = 0
    best_score = 0
    
    for i in range(len(content) - max_length):
        snippet = content_lower[i:i + max_length]
        score = sum(1 for word in query_words if word in snippet)
        if score > best_score:
            best_score = score
            best_pos = i
    
    # Extract snippet
    snippet = content[best_pos:best_pos + max_length]
    
    # Add ellipsis if needed
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_length < len(content):
        snippet = snippet + "..."
    
    return snippet
