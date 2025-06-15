"""LLM client implementations for various providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from anthropic import Anthropic, AsyncAnthropic

from ..core.config import settings
from ..core.constants import LLMProvider
from ..core.exceptions import LLMError

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "provider": self.__class__.__name__,
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4-turbo-preview",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        """
        super().__init__(model_name, max_tokens, temperature, **kwargs)
        
        self.api_key = api_key or settings.llm.openai_api_key
        if not self.api_key:
            raise LLMError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Merge parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.kwargs,
                **kwargs
            }
            
            # Generate response
            response = await self.client.chat.completions.create(**params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Merge parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                **self.kwargs,
                **kwargs
            }
            
            # Generate streaming response
            async for chunk in await self.client.chat.completions.create(**params):
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise LLMError(f"OpenAI streaming generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            import tiktoken
            
            # Get encoding for the model
            if "gpt-4" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except Exception:
            # Fallback: rough estimation
            return len(text.split()) * 1.3


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM client."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "claude-3-sonnet-20240229",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model_name: Claude model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        """
        super().__init__(model_name, max_tokens, temperature, **kwargs)
        
        self.api_key = api_key or settings.llm.anthropic_api_key
        if not self.api_key:
            raise LLMError("Anthropic API key is required")
        
        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using Anthropic API.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
                **self.kwargs,
                **kwargs
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            # Generate response
            response = await self.client.messages.create(**params)
            
            return response.content[0].text
            
        except Exception as e:
            raise LLMError(f"Anthropic generation failed: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using Anthropic API.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                **self.kwargs,
                **kwargs
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            # Generate streaming response
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise LLMError(f"Anthropic streaming generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens for Anthropic models.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens (rough estimation)
        """
        # Anthropic doesn't provide a public tokenizer
        # Use rough estimation: ~4 characters per token
        return len(text) // 4


class LocalLLMClient(BaseLLMClient):
    """Local LLM client for self-hosted models."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "local-model",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Initialize local LLM client.
        
        Args:
            model_path: Path to the local model
            model_name: Name identifier for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to run the model on
            **kwargs: Additional parameters
        """
        super().__init__(model_name, max_tokens, temperature, **kwargs)
        
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    async def _load_model(self) -> None:
        """Load the local model."""
        if self._loaded:
            return
        
        try:
            # This is a placeholder implementation
            # In practice, you would load your specific local model here
            # For example, using transformers, llama.cpp, or other frameworks
            
            logger.info(f"Loading local model from {self.model_path}")
            
            # Example with transformers (commented out as it requires specific setup)
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
            # self._model.to(self.device)
            
            self._loaded = True
            logger.info("Local model loaded successfully")
            
        except Exception as e:
            raise LLMError(f"Failed to load local model: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using local model.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        await self._load_model()
        
        try:
            # Placeholder implementation
            # In practice, you would implement generation using your local model
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # This is a mock response
            response = f"Local model response to: {full_prompt[:50]}..."
            
            return response
            
        except Exception as e:
            raise LLMError(f"Local model generation failed: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using local model.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        # For demonstration, yield the full response in chunks
        response = await self.generate(prompt, system_prompt, **kwargs)
        
        # Split into chunks
        chunk_size = 10
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    def count_tokens(self, text: str) -> int:
        """Count tokens for local model.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        # Simple word-based estimation
        return len(text.split())


class LLMClientFactory:
    """Factory for creating LLM client instances."""
    
    @staticmethod
    def create_client(
        provider: LLMProvider = None,
        **kwargs
    ) -> BaseLLMClient:
        """Create an LLM client instance.
        
        Args:
            provider: LLM provider type
            **kwargs: Additional parameters for the client
            
        Returns:
            LLM client instance
        """
        if provider is None:
            provider = settings.llm.provider
        
        if provider == LLMProvider.OPENAI:
            return OpenAIClient(
                api_key=settings.llm.openai_api_key,
                model_name=settings.llm.openai_model,
                max_tokens=settings.llm.openai_max_tokens,
                temperature=settings.llm.openai_temperature,
                **kwargs
            )
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                api_key=settings.llm.anthropic_api_key,
                model_name=settings.llm.anthropic_model,
                max_tokens=settings.llm.anthropic_max_tokens,
                **kwargs
            )
        elif provider == LLMProvider.LOCAL:
            return LocalLLMClient(
                model_path=settings.llm.local_model_path,
                device=settings.llm.local_model_device,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Global LLM client instance
llm_client = LLMClientFactory.create_client()
