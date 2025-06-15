"""Table embedding encoder using TAPAS and similar models."""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import TapasConfig, TapasModel, TapasTokenizer

from ..core.constants import Modality
from ..core.exceptions import EncodingError, ModelLoadError
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class TableEncoder(BaseEncoder):
    """Table embedding encoder using TAPAS model."""
    
    def __init__(
        self,
        model_name: str = "google/tapas-base",
        device: str = "cpu",
        batch_size: int = 4,  # Very small batch size for tables
        normalize: bool = True,
        max_seq_length: int = 512,
    ) -> None:
        """Initialize the table encoder.
        
        Args:
            model_name: Name of the TAPAS model
            device: Device to run the model on
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            max_seq_length: Maximum sequence length for tokenization
        """
        super().__init__(model_name, device, batch_size, normalize)
        self.max_seq_length = max_seq_length
    
    async def load_model(self) -> None:
        """Load the TAPAS model and tokenizer."""
        try:
            logger.info(f"Loading table encoder: {self.model_name}")
            
            # Load model and tokenizer in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            # Move model to device
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            
            # Set model to evaluation mode
            self._model.eval()
            
            self._is_loaded = True
            logger.info(f"Successfully loaded table encoder: {self.model_name}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load table encoder {self.model_name}: {e}")
    
    def _load_model_sync(self) -> tuple:
        """Synchronously load the model and tokenizer."""
        tokenizer = TapasTokenizer.from_pretrained(self.model_name)
        model = TapasModel.from_pretrained(self.model_name)
        return model, tokenizer
    
    async def encode(
        self,
        inputs: Union[Dict, pd.DataFrame, List[Union[Dict, pd.DataFrame]]],
        queries: Union[str, List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode table inputs into embeddings.
        
        Args:
            inputs: Table data as dict, DataFrame, or list of tables
            queries: Optional queries for table-question pairs
            **kwargs: Additional encoding parameters
            
        Returns:
            Numpy array of embeddings
        """
        await self.ensure_loaded()
        
        # Normalize inputs to list
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if not inputs:
            return np.array([])
        
        # Normalize queries
        if queries is None:
            queries = [""] * len(inputs)
        elif isinstance(queries, str):
            queries = [queries] * len(inputs)
        elif len(queries) != len(inputs):
            raise ValueError("Number of queries must match number of tables")
        
        try:
            # Convert inputs to standardized format
            processed_inputs = []
            for table, query in zip(inputs, queries):
                processed_table = self._process_table(table)
                processed_inputs.append((processed_table, query))
            
            # Process in batches
            if len(processed_inputs) <= self.batch_size:
                embeddings = await self._encode_batch(processed_inputs, **kwargs)
            else:
                all_embeddings = []
                batches = self._batch_inputs(processed_inputs)
                
                for batch in batches:
                    batch_embeddings = await self._encode_batch(batch, **kwargs)
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EncodingError(f"Failed to encode table inputs: {e}")
    
    def _encode_batch_sync(self, batch: List[tuple], kwargs: Dict) -> np.ndarray:
        """Synchronous batch encoding method.
        
        Args:
            batch: Batch of (table, query) tuples
            kwargs: Additional encoding parameters
            
        Returns:
            Batch embeddings
        """
        tables = []
        queries = []
        
        for table, query in batch:
            tables.append(table)
            queries.append(query)
        
        # Tokenize tables and queries
        inputs = self._tokenizer(
            table=tables,
            queries=queries,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            
            # Use pooler output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def _process_table(self, table: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Process table input into standardized DataFrame format.
        
        Args:
            table: Table data as dict or DataFrame
            
        Returns:
            Processed DataFrame
        """
        if isinstance(table, dict):
            # Convert dict to DataFrame
            if 'data' in table and 'columns' in table:
                # Format: {"data": [[row1], [row2], ...], "columns": [col1, col2, ...]}
                df = pd.DataFrame(table['data'], columns=table['columns'])
            else:
                # Format: {"col1": [val1, val2, ...], "col2": [val1, val2, ...]}
                df = pd.DataFrame(table)
        elif isinstance(table, pd.DataFrame):
            df = table.copy()
        else:
            raise ValueError(f"Unsupported table format: {type(table)}")
        
        # Clean and preprocess
        df = df.fillna("")  # Fill NaN values
        df = df.astype(str)  # Convert all to string for TAPAS
        
        # Limit table size to avoid memory issues
        max_rows = 50
        max_cols = 10
        
        if len(df) > max_rows:
            df = df.head(max_rows)
            logger.warning(f"Table truncated to {max_rows} rows")
        
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
            logger.warning(f"Table truncated to {max_cols} columns")
        
        return df
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this encoder.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return default dimensions for common TAPAS models
            model_dims = {
                "google/tapas-base": 768,
                "google/tapas-large": 1024,
                "google/tapas-base-finetuned-wtq": 768,
                "google/tapas-large-finetuned-wtq": 1024,
            }
            return model_dims.get(self.model_name, 768)
        
        return self._model.config.hidden_size
    
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles.
        
        Returns:
            Table modality
        """
        return Modality.TABLE
    
    def encode_with_questions(
        self,
        tables: List[Union[Dict, pd.DataFrame]],
        questions: List[str],
        **kwargs
    ) -> np.ndarray:
        """Encode tables with specific questions for better context.
        
        Args:
            tables: List of table data
            questions: List of questions about the tables
            **kwargs: Additional encoding parameters
            
        Returns:
            Table-question embeddings
        """
        return self.encode(tables, queries=questions, **kwargs)
    
    def encode_table_metadata(
        self,
        table: Union[Dict, pd.DataFrame],
        metadata: Dict = None,
        **kwargs
    ) -> np.ndarray:
        """Encode table with its metadata for richer representation.
        
        Args:
            table: Table data
            metadata: Additional metadata (title, description, etc.)
            **kwargs: Additional encoding parameters
            
        Returns:
            Enhanced table embedding
        """
        # Create a query from metadata
        query_parts = []
        
        if metadata:
            if 'title' in metadata:
                query_parts.append(f"Title: {metadata['title']}")
            if 'description' in metadata:
                query_parts.append(f"Description: {metadata['description']}")
            if 'source' in metadata:
                query_parts.append(f"Source: {metadata['source']}")
        
        query = " ".join(query_parts) if query_parts else ""
        
        return self.encode(table, queries=query, **kwargs)


class SimpleTableEncoder(BaseEncoder):
    """Simple table encoder that converts tables to text and uses text embeddings."""
    
    def __init__(
        self,
        text_encoder,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        """Initialize the simple table encoder.
        
        Args:
            text_encoder: Text encoder to use for table text
            device: Device to run on
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        super().__init__("simple-table-encoder", device, batch_size, normalize)
        self.text_encoder = text_encoder
    
    async def load_model(self) -> None:
        """Load the text encoder."""
        await self.text_encoder.ensure_loaded()
        self._is_loaded = True
    
    async def encode(
        self,
        inputs: Union[Dict, pd.DataFrame, List[Union[Dict, pd.DataFrame]]],
        **kwargs
    ) -> np.ndarray:
        """Encode tables by converting to text first.
        
        Args:
            inputs: Table data
            **kwargs: Additional encoding parameters
            
        Returns:
            Table embeddings
        """
        await self.ensure_loaded()
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Convert tables to text
        text_representations = []
        for table in inputs:
            text = self._table_to_text(table)
            text_representations.append(text)
        
        # Use text encoder
        return await self.text_encoder.encode(text_representations, **kwargs)
    
    def _table_to_text(self, table: Union[Dict, pd.DataFrame]) -> str:
        """Convert table to text representation.
        
        Args:
            table: Table data
            
        Returns:
            Text representation of the table
        """
        df = self._process_table(table)
        
        # Create text representation
        text_parts = []
        
        # Add column headers
        text_parts.append("Columns: " + ", ".join(df.columns))
        
        # Add rows
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(text_parts)
    
    def _process_table(self, table: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Process table input into DataFrame."""
        if isinstance(table, dict):
            if 'data' in table and 'columns' in table:
                df = pd.DataFrame(table['data'], columns=table['columns'])
            else:
                df = pd.DataFrame(table)
        elif isinstance(table, pd.DataFrame):
            df = table.copy()
        else:
            raise ValueError(f"Unsupported table format: {type(table)}")
        
        return df.fillna("").astype(str)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from text encoder."""
        return self.text_encoder.get_embedding_dimension()
    
    def get_modality(self) -> Modality:
        """Get the modality this encoder handles."""
        return Modality.TABLE
    
    def _encode_batch_sync(self, batch, kwargs):
        """Not used in this implementation."""
        pass
