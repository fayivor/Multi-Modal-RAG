"""Document management API routes."""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import List

import aiofiles
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...core.config import settings
from ...core.exceptions import IngestionError
from ...embeddings.multi_modal import multi_modal_system
from ...ingestion.pipeline import pipeline
from ...retrieval.vector_store import vector_store
from ..models.requests import BulkDocumentRequest, DocumentDeleteRequest, DocumentUploadRequest
from ..models.responses import BulkOperationResponse, DocumentListResponse, DocumentUploadResponse

logger = structlog.get_logger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    source: str = Form(None),
    category: str = Form(None),
    tags: str = Form("[]"),
    overwrite: bool = Form(False),
) -> DocumentUploadResponse:
    """Upload and process a document.
    
    Supports multiple file types including PDF, DOCX, text files, code files,
    and images. The document will be processed, chunked, embedded, and stored
    in the vector database.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info(
        "Document upload started",
        filename=file.filename,
        content_type=file.content_type,
        file_size=file.size,
        request_id=request_id,
    )
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Check file size
        if file.size and file.size > settings.processing.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file.size}) exceeds maximum allowed size ({settings.processing.max_file_size})"
            )
        
        # Check file extension
        file_path = Path(file.filename)
        if file_path.suffix.lower() not in settings.processing.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"File type {file_path.suffix} not supported"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create upload directory if it doesn't exist
        upload_dir = settings.processing.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_save_path = upload_dir / f"{document_id}_{file.filename}"
        
        async with aiofiles.open(file_save_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(
            "File saved to disk",
            save_path=str(file_save_path),
            request_id=request_id,
        )
        
        # Process the document
        processing_result = await pipeline.process_file(file_save_path)
        
        if processing_result.status != "completed":
            raise IngestionError(f"Document processing failed: {processing_result.error_message}")
        
        logger.info(
            "Document processing completed",
            documents_created=len(processing_result.documents),
            processing_time=processing_result.processing_time,
            request_id=request_id,
        )
        
        # Generate embeddings for all document chunks
        embedded_documents = []
        
        for doc in processing_result.documents:
            try:
                # Generate embeddings
                embeddings = await multi_modal_system.encode(
                    doc.content,
                    doc.modality,
                    align_to_unified_space=True
                )
                
                # Add embeddings to document
                doc.embeddings = embeddings.flatten().tolist()
                doc.chunk_id = doc.chunk_id or f"{document_id}_{len(embedded_documents)}"
                doc.parent_id = document_id
                
                # Add upload metadata
                doc.metadata.update({
                    "upload_source": source or "api",
                    "upload_category": category,
                    "upload_tags": tags.split(",") if tags else [],
                    "upload_timestamp": time.time(),
                    "document_id": document_id,
                })
                
                embedded_documents.append(doc)
                
            except Exception as e:
                logger.warning(
                    "Failed to embed document chunk",
                    chunk_id=doc.chunk_id,
                    error=str(e),
                    request_id=request_id,
                )
        
        if not embedded_documents:
            raise IngestionError("No document chunks could be embedded")
        
        # Store in vector database
        await vector_store.upsert(embedded_documents)
        
        logger.info(
            "Documents stored in vector database",
            chunks_stored=len(embedded_documents),
            request_id=request_id,
        )
        
        # Detect modality from first document
        detected_modality = embedded_documents[0].modality
        
        # Clean up temporary file
        try:
            file_save_path.unlink()
        except Exception as e:
            logger.warning(
                "Failed to clean up temporary file",
                file_path=str(file_save_path),
                error=str(e),
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file.size or len(content),
            modality=detected_modality,
            chunks_created=len(embedded_documents),
            processing_time_ms=processing_time_ms,
            status="completed",
            metadata={
                "source": source,
                "category": category,
                "tags": tags.split(",") if tags else [],
                "request_id": request_id,
            }
        )
        
        logger.info(
            "Document upload completed successfully",
            document_id=document_id,
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )
        
        return response
        
    except HTTPException:
        raise
    except IngestionError as e:
        logger.error(
            "Document ingestion error",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            "Unexpected error during document upload",
            error=str(e),
            request_id=request_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Document upload failed")


@router.delete("/documents", response_model=dict)
@limiter.limit("20/minute")
async def delete_documents(
    request: DocumentDeleteRequest,
    http_request: Request,
) -> dict:
    """Delete documents by their IDs."""
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    logger.info(
        "Document deletion requested",
        document_ids=request.document_ids,
        count=len(request.document_ids),
        request_id=request_id,
    )
    
    try:
        # Delete from vector store
        await vector_store.delete(request.document_ids)
        
        logger.info(
            "Documents deleted successfully",
            deleted_count=len(request.document_ids),
            request_id=request_id,
        )
        
        return {
            "message": f"Successfully deleted {len(request.document_ids)} documents",
            "deleted_ids": request.document_ids,
            "request_id": request_id,
        }
        
    except Exception as e:
        logger.error(
            "Error deleting documents",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Document deletion failed")


@router.post("/documents/bulk", response_model=BulkOperationResponse)
@limiter.limit("5/minute")
async def bulk_document_operation(
    request: BulkDocumentRequest,
    http_request: Request,
) -> BulkOperationResponse:
    """Perform bulk operations on documents."""
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    logger.info(
        "Bulk document operation started",
        operation=request.operation,
        document_count=len(request.documents),
        batch_size=request.batch_size,
        request_id=request_id,
    )
    
    try:
        successful = 0
        failed = 0
        errors = []
        batch_results = []
        
        # Process documents in batches
        for i in range(0, len(request.documents), request.batch_size):
            batch = request.documents[i:i + request.batch_size]
            batch_start_time = time.time()
            
            try:
                if request.operation == "delete":
                    # Extract document IDs from batch
                    doc_ids = [doc.get("id") for doc in batch if doc.get("id")]
                    if doc_ids:
                        await vector_store.delete(doc_ids)
                        successful += len(doc_ids)
                    
                elif request.operation == "upload":
                    # This would require file handling - simplified for now
                    logger.warning("Bulk upload not fully implemented")
                    failed += len(batch)
                    
                elif request.operation == "update":
                    # This would require updating document metadata
                    logger.warning("Bulk update not fully implemented")
                    failed += len(batch)
                
                batch_time = (time.time() - batch_start_time) * 1000
                batch_results.append({
                    "batch_index": i // request.batch_size,
                    "items_processed": len(batch),
                    "processing_time_ms": batch_time,
                    "status": "completed"
                })
                
            except Exception as e:
                failed += len(batch)
                error_detail = {
                    "batch_index": i // request.batch_size,
                    "error": str(e),
                    "items_affected": len(batch)
                }
                errors.append(error_detail)
                
                batch_results.append({
                    "batch_index": i // request.batch_size,
                    "items_processed": 0,
                    "processing_time_ms": 0,
                    "status": "failed",
                    "error": str(e)
                })
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = BulkOperationResponse(
            operation=request.operation,
            total_items=len(request.documents),
            successful=successful,
            failed=failed,
            errors=errors,
            processing_time_ms=processing_time_ms,
            batch_results=batch_results,
        )
        
        logger.info(
            "Bulk operation completed",
            operation=request.operation,
            successful=successful,
            failed=failed,
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error in bulk document operation",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Bulk operation failed")


@router.get("/documents", response_model=DocumentListResponse)
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    modality: str = None,
    source: str = None,
) -> DocumentListResponse:
    """List documents with pagination and filtering."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # This is a simplified implementation
        # In practice, you would query the vector store or a separate metadata store
        
        # For now, return a placeholder response
        documents = []
        total_count = 0
        
        # Apply filters
        filters_applied = {}
        if modality:
            filters_applied["modality"] = modality
        if source:
            filters_applied["source"] = source
        
        response = DocumentListResponse(
            documents=documents,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=False,
            filters_applied=filters_applied,
        )
        
        logger.info(
            "Document list retrieved",
            page=page,
            page_size=page_size,
            total_count=total_count,
            request_id=request_id,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error listing documents",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Failed to list documents")
