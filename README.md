# Multi-Modal RAG Technical Documentation Assistant

A production-ready RAG (Retrieval-Augmented Generation) system that can retrieve and reason over multiple data types including text, images, code, and tables to answer technical questions.

## Confidentiality Notice

**This repository represents a demonstration implementation that showcases the architectural methodologies, design patterns, and technical frameworks I employ in production enterprise environments.** While the core system design, implementation strategies, and engineering approaches are authentic representations of my professional work, specific business logic, proprietary algorithms, and client-sensitive details have been abstracted or reimplemented to comply with confidentiality agreements and intellectual property obligations.

## Features

- **Multi-Modal Support**: Process text documents, code repositories, images, diagrams, and tables
- **Advanced Retrieval**: Hybrid search combining vector similarity and BM25 with cross-encoder reranking
- **Production Ready**: Async processing, caching, monitoring, and comprehensive error handling
- **Scalable Architecture**: Designed for 1000+ concurrent users with efficient resource management
- **Flexible LLM Integration**: Support for OpenAI, Anthropic, and local models
- **Comprehensive API**: RESTful endpoints with proper versioning and documentation

## Architecture

```
multi-modal-rag/
├── src/
│   ├── core/           # Configuration and base classes
│   ├── ingestion/      # Document parsing and processing
│   ├── embeddings/     # Multi-modal embedding encoders
│   ├── retrieval/      # Vector search and reranking
│   ├── generation/     # LLM integration and response generation
│   ├── api/           # FastAPI application
│   └── utils/         # Logging, metrics, and utilities
├── tests/             # Comprehensive test suite
├── docker/            # Docker configuration
├── configs/           # Configuration files
└── docs/             # Documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- Redis (for caching)
- Vector database (Qdrant or Weaviate)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-modal-rag
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
uvicorn src.api.main:app --reload
```

### Docker Setup

```bash
docker-compose up -d
```

## Configuration

The system uses environment variables for configuration. Key settings include:

- `OPENAI_API_KEY`: OpenAI API key for LLM integration
- `QDRANT_HOST`: Vector database host
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

See `.env.example` for complete configuration options.

## API Usage

### Search Documents

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to implement async functions in Python?",
    "modalities": ["text", "code"],
    "top_k": 10
  }'
```

### Upload Documents

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@document.pdf" \
  -F "metadata={\"source\": \"manual\", \"category\": \"technical\"}"
```

## Development

### Code Style

This project follows strict coding standards:

- **Formatter**: Black (88 character line length)
- **Import sorting**: isort
- **Linting**: flake8
- **Type checking**: mypy
- **Testing**: pytest with >80% coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run integration tests only
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

## Performance

- **Search Latency**: <500ms for most queries
- **Throughput**: Designed for 1000+ concurrent users
- **Memory Efficient**: Lazy loading of ML models
- **Caching**: Multi-level caching strategy (embeddings, search results, LLM responses)

## Monitoring

The system includes comprehensive monitoring:

- **Metrics**: Prometheus metrics for all critical operations
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Built-in health check endpoints
- **Error Tracking**: Integration with Sentry for error monitoring

## Security

- **Input Validation**: Comprehensive validation using Pydantic
- **Rate Limiting**: Per-user rate limiting to prevent abuse
- **File Security**: Secure file handling with size and type restrictions
- **Authentication**: JWT-based authentication (optional)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass and coverage is maintained
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:

- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [FAQ](docs/faq.md)
