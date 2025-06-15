"""Citation tracking and source attribution for generated responses."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .prompt_builder import ContextDocument

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """Individual citation with source information."""
    
    id: str = Field(..., description="Citation identifier")
    source: str = Field(..., description="Source document or file")
    content_snippet: str = Field(..., description="Relevant content snippet")
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    url: Optional[str] = None
    metadata: Dict[str, any] = Field(default_factory=dict, description="Additional metadata")


class CitationMap(BaseModel):
    """Mapping of citation IDs to citation information."""
    
    citations: Dict[str, Citation] = Field(default_factory=dict, description="Citation mapping")
    used_citations: Set[str] = Field(default_factory=set, description="Citations used in response")
    
    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the map.
        
        Args:
            citation: Citation to add
        """
        self.citations[citation.id] = citation
    
    def mark_used(self, citation_id: str) -> None:
        """Mark a citation as used.
        
        Args:
            citation_id: ID of the citation to mark as used
        """
        if citation_id in self.citations:
            self.used_citations.add(citation_id)
    
    def get_used_citations(self) -> List[Citation]:
        """Get list of used citations.
        
        Returns:
            List of used citations
        """
        return [
            self.citations[cid] for cid in self.used_citations
            if cid in self.citations
        ]
    
    def get_unused_citations(self) -> List[Citation]:
        """Get list of unused citations.
        
        Returns:
            List of unused citations
        """
        return [
            citation for cid, citation in self.citations.items()
            if cid not in self.used_citations
        ]


class CitationTracker:
    """Tracker for managing citations in generated responses."""
    
    def __init__(self) -> None:
        """Initialize the citation tracker."""
        self.citation_patterns = [
            r'\[doc_(\d+)\]',           # [doc_1], [doc_2], etc.
            r'\[(\d+)\]',               # [1], [2], etc.
            r'\(doc_(\d+)\)',           # (doc_1), (doc_2), etc.
            r'\((\d+)\)',               # (1), (2), etc.
            r'\[source_(\d+)\]',        # [source_1], [source_2], etc.
            r'\[ref_(\d+)\]',           # [ref_1], [ref_2], etc.
        ]
    
    def create_citation_map(self, context_documents: List[ContextDocument]) -> CitationMap:
        """Create a citation map from context documents.
        
        Args:
            context_documents: List of context documents
            
        Returns:
            Citation map
        """
        citation_map = CitationMap()
        
        for doc in context_documents:
            # Create citation
            citation = Citation(
                id=doc.citation_id,
                source=doc.source,
                content_snippet=self._create_snippet(doc.content),
                metadata=doc.metadata.copy()
            )
            
            # Add page number if available
            if 'page_number' in doc.metadata:
                citation.page_number = doc.metadata['page_number']
            
            # Add line number if available
            if 'start_line' in doc.metadata:
                citation.line_number = doc.metadata['start_line']
            
            # Add URL if available
            if 'url' in doc.metadata:
                citation.url = doc.metadata['url']
            
            citation_map.add_citation(citation)
        
        return citation_map
    
    def extract_citations_from_response(
        self,
        response: str,
        citation_map: CitationMap
    ) -> Tuple[str, List[Citation]]:
        """Extract citations from a generated response.
        
        Args:
            response: Generated response text
            citation_map: Citation map to use for lookup
            
        Returns:
            Tuple of (cleaned response, list of used citations)
        """
        used_citation_ids = set()
        
        # Find all citation patterns in the response
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Extract citation ID
                if match.groups():
                    citation_id = match.group(1)
                    # Normalize citation ID format
                    if not citation_id.startswith('doc_'):
                        citation_id = f"doc_{citation_id}"
                    used_citation_ids.add(citation_id)
        
        # Mark citations as used
        for citation_id in used_citation_ids:
            citation_map.mark_used(citation_id)
        
        # Get used citations
        used_citations = citation_map.get_used_citations()
        
        return response, used_citations
    
    def format_citations(
        self,
        citations: List[Citation],
        format_style: str = "numbered"
    ) -> str:
        """Format citations for display.
        
        Args:
            citations: List of citations to format
            format_style: Citation format style (numbered, apa, mla)
            
        Returns:
            Formatted citations string
        """
        if not citations:
            return ""
        
        if format_style == "numbered":
            return self._format_numbered_citations(citations)
        elif format_style == "apa":
            return self._format_apa_citations(citations)
        elif format_style == "mla":
            return self._format_mla_citations(citations)
        else:
            return self._format_numbered_citations(citations)
    
    def _format_numbered_citations(self, citations: List[Citation]) -> str:
        """Format citations in numbered style.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citations
        """
        formatted = ["## Sources"]
        
        for i, citation in enumerate(citations, 1):
            # Extract number from citation ID
            citation_num = citation.id.replace('doc_', '')
            
            parts = [f"{citation_num}. {citation.source}"]
            
            if citation.page_number:
                parts.append(f"Page {citation.page_number}")
            
            if citation.line_number:
                parts.append(f"Line {citation.line_number}")
            
            if citation.url:
                parts.append(f"URL: {citation.url}")
            
            # Add content snippet
            if citation.content_snippet:
                parts.append(f"Excerpt: \"{citation.content_snippet}\"")
            
            formatted.append(" | ".join(parts))
        
        return "\n".join(formatted)
    
    def _format_apa_citations(self, citations: List[Citation]) -> str:
        """Format citations in APA style.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citations
        """
        formatted = ["## References"]
        
        for citation in citations:
            # Basic APA format: Author. (Year). Title. Source.
            parts = []
            
            # Extract metadata
            title = citation.metadata.get('title', citation.source)
            author = citation.metadata.get('author', 'Unknown Author')
            year = citation.metadata.get('year', 'n.d.')
            
            apa_citation = f"{author}. ({year}). {title}."
            
            if citation.url:
                apa_citation += f" Retrieved from {citation.url}"
            
            formatted.append(apa_citation)
        
        return "\n".join(formatted)
    
    def _format_mla_citations(self, citations: List[Citation]) -> str:
        """Format citations in MLA style.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citations
        """
        formatted = ["## Works Cited"]
        
        for citation in citations:
            # Basic MLA format: Author. "Title." Source, Date.
            parts = []
            
            # Extract metadata
            title = citation.metadata.get('title', citation.source)
            author = citation.metadata.get('author', 'Unknown Author')
            date = citation.metadata.get('date', 'n.d.')
            
            mla_citation = f'{author}. "{title}." {citation.source}, {date}.'
            
            if citation.url:
                mla_citation += f" Web. {citation.url}"
            
            formatted.append(mla_citation)
        
        return "\n".join(formatted)
    
    def _create_snippet(self, content: str, max_length: int = 150) -> str:
        """Create a content snippet for citation.
        
        Args:
            content: Full content
            max_length: Maximum snippet length
            
        Returns:
            Content snippet
        """
        if len(content) <= max_length:
            return content
        
        # Try to break at sentence boundary
        sentences = content.split('. ')
        snippet = ""
        
        for sentence in sentences:
            if len(snippet + sentence) <= max_length - 3:
                snippet += sentence + ". "
            else:
                break
        
        if snippet:
            return snippet.strip()
        
        # Fallback: truncate at word boundary
        words = content.split()
        snippet = ""
        
        for word in words:
            if len(snippet + word) <= max_length - 3:
                snippet += word + " "
            else:
                break
        
        return snippet.strip() + "..."
    
    def validate_citations(
        self,
        response: str,
        citation_map: CitationMap
    ) -> Dict[str, any]:
        """Validate citations in a response.
        
        Args:
            response: Generated response
            citation_map: Citation map
            
        Returns:
            Validation results
        """
        # Extract citations from response
        _, used_citations = self.extract_citations_from_response(response, citation_map)
        
        # Find citation references in text
        citation_refs = set()
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    citation_id = match.group(1)
                    if not citation_id.startswith('doc_'):
                        citation_id = f"doc_{citation_id}"
                    citation_refs.add(citation_id)
        
        # Check for invalid citations
        invalid_citations = []
        for citation_id in citation_refs:
            if citation_id not in citation_map.citations:
                invalid_citations.append(citation_id)
        
        # Check for missing citations (claims without citations)
        # This is a simplified check - in practice, you might use NLP to detect claims
        sentences = response.split('.')
        uncited_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Ignore very short sentences
                has_citation = any(
                    re.search(pattern, sentence, re.IGNORECASE)
                    for pattern in self.citation_patterns
                )
                if not has_citation:
                    uncited_sentences.append(sentence)
        
        return {
            "total_citations": len(citation_map.citations),
            "used_citations": len(used_citations),
            "unused_citations": len(citation_map.get_unused_citations()),
            "invalid_citations": invalid_citations,
            "citation_coverage": len(used_citations) / len(citation_map.citations) if citation_map.citations else 0,
            "uncited_sentences_count": len(uncited_sentences),
            "validation_passed": len(invalid_citations) == 0,
        }
    
    def add_missing_citations(
        self,
        response: str,
        citation_map: CitationMap,
        threshold: float = 0.7
    ) -> str:
        """Add citations to sentences that likely need them.
        
        Args:
            response: Generated response
            citation_map: Citation map
            threshold: Similarity threshold for adding citations
            
        Returns:
            Response with added citations
        """
        # This is a simplified implementation
        # In practice, you might use semantic similarity to match sentences to sources
        
        sentences = response.split('.')
        modified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence already has citation
            has_citation = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in self.citation_patterns
            )
            
            if not has_citation and len(sentence) > 20:
                # Try to find relevant citation
                best_citation = self._find_best_citation(sentence, citation_map)
                if best_citation:
                    sentence += f" [{best_citation.id}]"
            
            modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences)
    
    def _find_best_citation(
        self,
        sentence: str,
        citation_map: CitationMap
    ) -> Optional[Citation]:
        """Find the best citation for a sentence.
        
        Args:
            sentence: Sentence to find citation for
            citation_map: Citation map
            
        Returns:
            Best matching citation or None
        """
        # Simple keyword-based matching
        # In practice, you might use semantic similarity
        
        sentence_words = set(sentence.lower().split())
        best_citation = None
        best_score = 0
        
        for citation in citation_map.citations.values():
            content_words = set(citation.content_snippet.lower().split())
            overlap = len(sentence_words.intersection(content_words))
            score = overlap / len(sentence_words) if sentence_words else 0
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_score = score
                best_citation = citation
        
        return best_citation


# Global citation tracker instance
citation_tracker = CitationTracker()
