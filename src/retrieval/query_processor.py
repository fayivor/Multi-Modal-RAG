"""Query processing and intent analysis for retrieval."""

import logging
import re
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..core.constants import Modality, QueryIntent
from ..core.exceptions import QueryProcessingError

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    
    original_query: str = Field(..., description="Original query text")
    processed_query: str = Field(..., description="Processed query text")
    intent: QueryIntent = Field(..., description="Detected query intent")
    modalities: List[Modality] = Field(..., description="Relevant modalities")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    filters: Dict[str, any] = Field(default_factory=dict, description="Suggested filters")
    confidence: float = Field(default=0.0, description="Analysis confidence score")
    metadata: Dict[str, any] = Field(default_factory=dict, description="Additional metadata")


class QueryProcessor:
    """Processor for analyzing and enhancing search queries."""
    
    def __init__(self) -> None:
        """Initialize the query processor."""
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.CODE_SEARCH: [
                r'\b(function|class|method|variable|import|def|return)\b',
                r'\b(python|javascript|java|cpp|code|programming)\b',
                r'\b(implement|algorithm|syntax|debug|error)\b',
                r'[{}()\[\];]',  # Code symbols
            ],
            QueryIntent.VISUAL_SEARCH: [
                r'\b(image|diagram|chart|graph|figure|screenshot)\b',
                r'\b(visual|picture|illustration|drawing)\b',
                r'\b(architecture|flowchart|uml|network)\b',
            ],
            QueryIntent.FACTUAL_SEARCH: [
                r'\b(what|who|when|where|why|how)\b',
                r'\b(definition|meaning|explain|describe)\b',
                r'\b(fact|information|detail|specification)\b',
            ],
            QueryIntent.DEBUG_SEARCH: [
                r'\b(error|bug|issue|problem|fix|solve)\b',
                r'\b(debug|troubleshoot|exception|crash)\b',
                r'\b(not working|failed|broken)\b',
            ],
        }
        
        # Modality detection patterns
        self.modality_patterns = {
            Modality.CODE: [
                r'\b(code|function|class|method|variable|algorithm)\b',
                r'\b(python|javascript|java|cpp|programming)\b',
                r'\b(syntax|implementation|debug)\b',
            ],
            Modality.IMAGE: [
                r'\b(image|diagram|chart|graph|figure)\b',
                r'\b(visual|picture|illustration)\b',
                r'\b(architecture|flowchart|screenshot)\b',
            ],
            Modality.TABLE: [
                r'\b(table|data|column|row|csv|spreadsheet)\b',
                r'\b(statistics|numbers|values|dataset)\b',
            ],
            Modality.TEXT: [
                r'\b(document|text|article|paragraph|content)\b',
                r'\b(documentation|manual|guide|tutorial)\b',
            ],
        }
        
        # Programming language patterns
        self.language_patterns = {
            'python': r'\b(python|py|def|import|class|self)\b',
            'javascript': r'\b(javascript|js|function|var|let|const)\b',
            'java': r'\b(java|public|private|class|static)\b',
            'cpp': r'\b(cpp|c\+\+|#include|namespace|std)\b',
            'sql': r'\b(sql|select|from|where|join|table)\b',
        }
        
        # Stop words for keyword extraction
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'how', 'what', 'when', 'where', 'why'
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a search query.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Query analysis result
        """
        try:
            # Clean and preprocess query
            processed_query = self._preprocess_query(query)
            
            # Detect intent
            intent, intent_confidence = self._detect_intent(processed_query)
            
            # Detect relevant modalities
            modalities = self._detect_modalities(processed_query)
            
            # Extract keywords
            keywords = self._extract_keywords(processed_query)
            
            # Extract entities
            entities = self._extract_entities(processed_query)
            
            # Generate filters
            filters = self._generate_filters(processed_query, intent, modalities)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                intent_confidence, modalities, keywords
            )
            
            # Additional metadata
            metadata = {
                'query_length': len(query),
                'word_count': len(query.split()),
                'has_code_symbols': bool(re.search(r'[{}()\[\];]', query)),
                'detected_languages': self._detect_programming_languages(query),
            }
            
            return QueryAnalysis(
                original_query=query,
                processed_query=processed_query,
                intent=intent,
                modalities=modalities,
                keywords=keywords,
                entities=entities,
                filters=filters,
                confidence=confidence,
                metadata=metadata,
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to analyze query: {e}")
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query text.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Convert to lowercase
        processed = query.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Handle special characters in code queries
        # Keep important code symbols but normalize others
        processed = re.sub(r'[^\w\s{}()\[\];.,!?-]', ' ', processed)
        
        return processed
    
    def _detect_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Detect the intent of the query.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            Tuple of (detected intent, confidence score)
        """
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            # Normalize by pattern count
            intent_scores[intent] = score / len(patterns)
        
        # Find the intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            
            # If no clear intent, default to general search
            if confidence < 0.1:
                return QueryIntent.GENERAL_SEARCH, 0.5
            
            return best_intent, min(confidence, 1.0)
        
        return QueryIntent.GENERAL_SEARCH, 0.5
    
    def _detect_modalities(self, query: str) -> List[Modality]:
        """Detect relevant modalities for the query.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            List of relevant modalities
        """
        modality_scores = {}
        
        for modality, patterns in self.modality_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            modality_scores[modality] = score
        
        # Select modalities with positive scores
        relevant_modalities = [
            modality for modality, score in modality_scores.items()
            if score > 0
        ]
        
        # If no specific modalities detected, include text by default
        if not relevant_modalities:
            relevant_modalities = [Modality.TEXT]
        
        # Sort by relevance score
        relevant_modalities.sort(
            key=lambda m: modality_scores[m], reverse=True
        )
        
        return relevant_modalities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from the query.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:10]  # Limit to top 10 keywords
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract programming languages
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                entities.append(lang)
        
        # Extract file extensions
        file_extensions = re.findall(r'\.\w{2,4}\b', query)
        entities.extend(file_extensions)
        
        # Extract quoted strings (potential exact matches)
        quoted_strings = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_strings)
        
        # Extract camelCase/PascalCase identifiers
        camel_case = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', query)
        entities.extend(camel_case)
        
        return list(set(entities))  # Remove duplicates
    
    def _generate_filters(
        self,
        query: str,
        intent: QueryIntent,
        modalities: List[Modality]
    ) -> Dict[str, any]:
        """Generate suggested filters based on query analysis.
        
        Args:
            query: Preprocessed query text
            intent: Detected intent
            modalities: Relevant modalities
            
        Returns:
            Dictionary of suggested filters
        """
        filters = {}
        
        # Modality filter
        if len(modalities) == 1:
            filters['modality'] = modalities[0]
        elif len(modalities) > 1:
            filters['modalities'] = modalities
        
        # Programming language filter
        detected_languages = self._detect_programming_languages(query)
        if detected_languages:
            filters['programming_language'] = detected_languages[0]
        
        # File type filter based on extensions
        file_extensions = re.findall(r'\.\w{2,4}\b', query)
        if file_extensions:
            filters['file_extension'] = file_extensions[0]
        
        # Date-based filters for recent/old content
        if re.search(r'\b(recent|latest|new|current)\b', query, re.IGNORECASE):
            filters['date_preference'] = 'recent'
        elif re.search(r'\b(old|legacy|deprecated|archive)\b', query, re.IGNORECASE):
            filters['date_preference'] = 'old'
        
        return filters
    
    def _detect_programming_languages(self, query: str) -> List[str]:
        """Detect programming languages mentioned in the query.
        
        Args:
            query: Query text
            
        Returns:
            List of detected programming languages
        """
        detected = []
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                detected.append(lang)
        return detected
    
    def _calculate_confidence(
        self,
        intent_confidence: float,
        modalities: List[Modality],
        keywords: List[str]
    ) -> float:
        """Calculate overall confidence score for the analysis.
        
        Args:
            intent_confidence: Confidence in intent detection
            modalities: Detected modalities
            keywords: Extracted keywords
            
        Returns:
            Overall confidence score (0-1)
        """
        # Base confidence from intent detection
        confidence = intent_confidence * 0.4
        
        # Add confidence based on modality detection
        if modalities:
            modality_confidence = min(len(modalities) * 0.2, 0.3)
            confidence += modality_confidence
        
        # Add confidence based on keyword extraction
        if keywords:
            keyword_confidence = min(len(keywords) * 0.05, 0.3)
            confidence += keyword_confidence
        
        return min(confidence, 1.0)
    
    def enhance_query(self, analysis: QueryAnalysis) -> str:
        """Enhance the query based on analysis results.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Enhanced query string
        """
        enhanced_parts = [analysis.processed_query]
        
        # Add relevant keywords
        if analysis.keywords:
            # Add top keywords that aren't already in the query
            query_words = set(analysis.processed_query.split())
            new_keywords = [
                kw for kw in analysis.keywords[:3]
                if kw not in query_words
            ]
            if new_keywords:
                enhanced_parts.extend(new_keywords)
        
        # Add entities
        if analysis.entities:
            # Add entities that might improve search
            query_text = analysis.processed_query
            new_entities = [
                entity for entity in analysis.entities[:2]
                if entity not in query_text
            ]
            if new_entities:
                enhanced_parts.extend(new_entities)
        
        return " ".join(enhanced_parts)


# Global query processor instance
query_processor = QueryProcessor()
