"""Prompt building and context formatting for LLM generation."""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.constants import MAX_CONTEXT_TOKENS, Modality, QueryIntent
from ..retrieval.query_processor import QueryAnalysis
from ..retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """Template for building prompts."""
    
    name: str = Field(..., description="Template name")
    system_prompt: str = Field(..., description="System prompt template")
    user_prompt: str = Field(..., description="User prompt template")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    max_context_length: int = Field(default=MAX_CONTEXT_TOKENS, description="Maximum context length")


class ContextDocument(BaseModel):
    """Document in the context with formatting."""
    
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source")
    modality: Modality = Field(..., description="Document modality")
    relevance_score: float = Field(..., description="Relevance score")
    citation_id: str = Field(..., description="Citation identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PromptContext(BaseModel):
    """Context for prompt building."""
    
    query: str = Field(..., description="User query")
    documents: List[ContextDocument] = Field(default_factory=list, description="Context documents")
    query_analysis: Optional[QueryAnalysis] = None
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    max_tokens: int = Field(default=MAX_CONTEXT_TOKENS, description="Maximum context tokens")


class PromptBuilder:
    """Builder for creating prompts with context formatting."""
    
    def __init__(self, llm_client=None) -> None:
        """Initialize the prompt builder.
        
        Args:
            llm_client: LLM client for token counting
        """
        self.llm_client = llm_client
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates."""
        templates = {}
        
        # General RAG template
        templates["general"] = PromptTemplate(
            name="general",
            system_prompt="""You are a helpful AI assistant that answers questions based on provided context documents. 

Instructions:
1. Answer the question using ONLY the information provided in the context documents
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite your sources using the format [doc_id] after each claim
4. Be precise and factual in your responses
5. If multiple documents provide conflicting information, acknowledge this

Context Documents:
{context}""",
            user_prompt="""Question: {query}

Please provide a comprehensive answer based on the context documents above.""",
            variables=["context", "query"]
        )
        
        # Code-specific template
        templates["code"] = PromptTemplate(
            name="code",
            system_prompt="""You are an expert programming assistant that helps with code-related questions using provided context.

Instructions:
1. Provide accurate code examples and explanations based on the context
2. Include proper syntax highlighting and formatting
3. Explain the code logic and any important concepts
4. Cite sources using [doc_id] format
5. If the context doesn't contain relevant code examples, state this clearly

Code Context:
{context}""",
            user_prompt="""Programming Question: {query}

Please provide a detailed answer with code examples if available in the context.""",
            variables=["context", "query"]
        )
        
        # Visual/diagram template
        templates["visual"] = PromptTemplate(
            name="visual",
            system_prompt="""You are an AI assistant specialized in explaining visual content, diagrams, and images.

Instructions:
1. Describe visual elements based on the provided context
2. Explain relationships and flows shown in diagrams
3. Reference specific visual elements when possible
4. Cite sources using [doc_id] format
5. If visual context is insufficient, request more specific information

Visual Context:
{context}""",
            user_prompt="""Visual Question: {query}

Please explain based on the visual information provided in the context.""",
            variables=["context", "query"]
        )
        
        # Debug/troubleshooting template
        templates["debug"] = PromptTemplate(
            name="debug",
            system_prompt="""You are a debugging expert that helps solve technical problems using provided context.

Instructions:
1. Analyze the problem systematically
2. Provide step-by-step troubleshooting guidance
3. Reference relevant error messages or symptoms from context
4. Suggest specific solutions based on the context
5. Cite sources using [doc_id] format

Debugging Context:
{context}""",
            user_prompt="""Problem/Error: {query}

Please provide debugging guidance based on the context information.""",
            variables=["context", "query"]
        )
        
        return templates
    
    def build_prompt(
        self,
        context: PromptContext,
        template_name: str = "general",
        custom_template: Optional[PromptTemplate] = None,
    ) -> tuple[str, str]:
        """Build a prompt from context and template.
        
        Args:
            context: Prompt context with query and documents
            template_name: Name of the template to use
            custom_template: Custom template to use instead
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Select template
        if custom_template:
            template = custom_template
        elif template_name in self.templates:
            template = self.templates[template_name]
        else:
            logger.warning(f"Template {template_name} not found, using general template")
            template = self.templates["general"]
        
        # Format context documents
        formatted_context = self._format_context_documents(
            context.documents, template.max_context_length
        )
        
        # Prepare template variables
        variables = {
            "context": formatted_context,
            "query": context.query,
        }
        
        # Add additional context variables
        variables.update(context.additional_context)
        
        # Format prompts
        try:
            system_prompt = template.system_prompt.format(**variables)
            user_prompt = template.user_prompt.format(**variables)
            
            return system_prompt, user_prompt
            
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
    
    def _format_context_documents(
        self,
        documents: List[ContextDocument],
        max_tokens: int
    ) -> str:
        """Format context documents for inclusion in prompt.
        
        Args:
            documents: List of context documents
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context documents found."
        
        formatted_docs = []
        current_tokens = 0
        
        for doc in documents:
            # Format document
            doc_text = self._format_single_document(doc)
            
            # Count tokens
            doc_tokens = self._count_tokens(doc_text)
            
            # Check if adding this document would exceed limit
            if current_tokens + doc_tokens > max_tokens:
                if not formatted_docs:
                    # If first document is too long, truncate it
                    truncated_doc = self._truncate_document(doc, max_tokens)
                    formatted_docs.append(truncated_doc)
                break
            
            formatted_docs.append(doc_text)
            current_tokens += doc_tokens
        
        return "\n\n".join(formatted_docs)
    
    def _format_single_document(self, doc: ContextDocument) -> str:
        """Format a single document for context.
        
        Args:
            doc: Context document to format
            
        Returns:
            Formatted document string
        """
        # Create header with metadata
        header_parts = [f"Document ID: {doc.citation_id}"]
        
        if doc.source:
            header_parts.append(f"Source: {doc.source}")
        
        header_parts.append(f"Type: {doc.modality.value}")
        header_parts.append(f"Relevance: {doc.relevance_score:.3f}")
        
        # Add additional metadata
        if doc.metadata:
            for key, value in doc.metadata.items():
                if key in ["title", "filename", "page_number", "function_name", "class_name"]:
                    header_parts.append(f"{key.title()}: {value}")
        
        header = " | ".join(header_parts)
        
        # Format content based on modality
        if doc.modality == Modality.CODE:
            content = f"```\n{doc.content}\n```"
        elif doc.modality == Modality.TABLE:
            content = f"Table Data:\n{doc.content}"
        elif doc.modality == Modality.IMAGE:
            content = f"Image Description: {doc.content}"
        else:
            content = doc.content
        
        return f"[{header}]\n{content}"
    
    def _truncate_document(self, doc: ContextDocument, max_tokens: int) -> str:
        """Truncate a document to fit within token limit.
        
        Args:
            doc: Document to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated document string
        """
        # Start with full formatting
        full_doc = self._format_single_document(doc)
        
        # If it fits, return as is
        if self._count_tokens(full_doc) <= max_tokens:
            return full_doc
        
        # Truncate content while preserving header
        lines = full_doc.split('\n')
        header_lines = []
        content_lines = []
        
        # Separate header and content
        in_content = False
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                header_lines.append(line)
            else:
                in_content = True
                if in_content:
                    content_lines.append(line)
        
        # Rebuild with truncated content
        header = '\n'.join(header_lines)
        header_tokens = self._count_tokens(header)
        
        available_tokens = max_tokens - header_tokens - 10  # Buffer
        
        truncated_content = []
        current_tokens = 0
        
        for line in content_lines:
            line_tokens = self._count_tokens(line)
            if current_tokens + line_tokens > available_tokens:
                break
            truncated_content.append(line)
            current_tokens += line_tokens
        
        if truncated_content:
            content = '\n'.join(truncated_content) + "\n... [truncated]"
        else:
            content = "[Content too long to display]"
        
        return f"{header}\n{content}"
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.llm_client:
            try:
                return self.llm_client.count_tokens(text)
            except Exception:
                pass
        
        # Fallback: rough estimation
        return len(text.split()) * 1.3
    
    def select_template_by_intent(self, intent: QueryIntent) -> str:
        """Select appropriate template based on query intent.
        
        Args:
            intent: Detected query intent
            
        Returns:
            Template name
        """
        intent_template_map = {
            QueryIntent.CODE_SEARCH: "code",
            QueryIntent.VISUAL_SEARCH: "visual",
            QueryIntent.DEBUG_SEARCH: "debug",
            QueryIntent.FACTUAL_SEARCH: "general",
            QueryIntent.GENERAL_SEARCH: "general",
        }
        
        return intent_template_map.get(intent, "general")
    
    def create_context_from_search_results(
        self,
        query: str,
        search_results: List[SearchResult],
        query_analysis: Optional[QueryAnalysis] = None,
        max_documents: int = 10,
    ) -> PromptContext:
        """Create prompt context from search results.
        
        Args:
            query: User query
            search_results: Search results to include
            query_analysis: Optional query analysis
            max_documents: Maximum number of documents to include
            
        Returns:
            Prompt context
        """
        # Convert search results to context documents
        context_docs = []
        
        for i, result in enumerate(search_results[:max_documents]):
            # Generate citation ID
            citation_id = f"doc_{i+1}"
            
            # Extract source information
            source = result.document.metadata.get('source', 'Unknown')
            if 'filename' in result.document.metadata:
                source = result.document.metadata['filename']
            
            context_doc = ContextDocument(
                content=result.document.content,
                source=source,
                modality=result.document.modality,
                relevance_score=result.score,
                citation_id=citation_id,
                metadata=result.document.metadata,
            )
            context_docs.append(context_doc)
        
        return PromptContext(
            query=query,
            documents=context_docs,
            query_analysis=query_analysis,
            max_tokens=settings.llm.openai_max_tokens or MAX_CONTEXT_TOKENS,
        )
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a custom template.
        
        Args:
            template: Template to add
        """
        self.templates[template.name] = template
        logger.info(f"Added custom template: {template.name}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())


# Global prompt builder instance
prompt_builder = PromptBuilder()
