# Vector Search Library - Production-Ready API Design

```python
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
from pathlib import Path

from vectorkit import VectorSearchClient
from vectorkit.databases import (
    QdrantDB, PineconeDB, WeaviateDB, ChromaDB, MilvusDB
)
from vectorkit.chunkers import (
    RecursiveTextSplitter, SentenceChunker, TokenChunker,
    SemanticChunker, MarkdownChunker, CodeChunker
)
from vectorkit.embedders import (
    HuggingFaceEmbedder, OpenAIEmbedder, CohereEmbedder,
    VoyageEmbedder, AnthropicEmbedder
)
from vectorkit.rerankers import (
    CrossEncoderReranker, CohereReranker, LLMReranker,
    ColBERTReranker
)
from vectorkit.retrievers import (
    VectorRetriever, HybridRetriever, KeywordRetriever,
    MMRRetriever, ContextualCompressionRetriever
)
from vectorkit.types import (
    Document, SearchResult, Collection, BatchResult,
    IngestionResult, SearchMetrics, CollectionStats
)
from vectorkit.exceptions import (
    VectorSearchError, CollectionNotFoundError,
    EmbeddingError, DatabaseConnectionError,
    ValidationError, RateLimitError
)

# ==============================================================================
# 1. INITIALIZATION WITH MULTIPLE PATTERNS
# ==============================================================================

# Pattern 1: Fluent Builder with Type Safety
client = (
    VectorSearchClient()
    .with_database(
        QdrantDB(
            url="http://localhost:6333",  # Support both host/port and URL
            api_key="${QDRANT_API_KEY}",  # Environment variable support
            timeout=30,
            grpc_port=6334,
            prefer_grpc=True,
            connection_pool_size=10,
            retry_config={
                "max_retries": 3,
                "backoff_factor": 2.0
            }
        )
    )
    .with_chunker(
        RecursiveTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,  # Can use tiktoken for token counting
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True
        )
    )
    .with_embedder(
        HuggingFaceEmbedder(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            cache_folder="./embeddings_cache",
            model_kwargs={"trust_remote_code": True}
        )
    )
    .with_retriever(
        HybridRetriever(
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=20,
            diversity=0.3,  # MMR diversity
            fusion_algorithm="reciprocal_rank"  # or "linear", "log"
        )
    )
    .with_reranker(
        CrossEncoderReranker(
            model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="auto",
            batch_size=16,
            top_k=10,
            cache_embeddings=True
        )
    )
    .with_cache(
        cache_type="redis",
        host="localhost",
        port=6379,
        ttl=3600,
        max_cache_size_mb=1024
    )
    .build()
)

# Pattern 2: Configuration-based (YAML/JSON/TOML)
client = VectorSearchClient.from_config("config.yaml")

# Pattern 3: Environment-based Auto-configuration
client = VectorSearchClient.from_env()  # Reads from env vars

# Pattern 4: Context Manager with Async Support
async def main():
    async with VectorSearchClient() as client:
        await client.configure_database(QdrantDB(url="http://localhost:6333"))
        await client.configure_embedder(HuggingFaceEmbedder(model_id="all-MiniLM-L6-v2"))
        # Auto cleanup on exit

# ==============================================================================
# 2. COLLECTION MANAGEMENT WITH SCHEMA VALIDATION
# ==============================================================================

# Create collection with full schema definition
collection = await client.collections.create(
    name="technical_documents",
    vector_config={
        "size": 384,  # Auto-detected from embedder if not specified
        "distance": "cosine",  # "euclidean", "dot_product"
        "on_disk": True,  # For large collections
        "quantization": {
            "type": "scalar",
            "quantile": 0.99,
            "always_ram": True
        }
    },
    schema={
        "title": {"type": "text", "tokenizer": "standard"},
        "category": {"type": "keyword", "index": True},
        "tags": {"type": "keyword[]", "index": True},
        "created_at": {"type": "datetime", "format": "iso8601"},
        "source": {"type": "text"},
        "page_number": {"type": "integer"},
        "author": {"type": "keyword"},
        "language": {"type": "keyword", "default": "en"}
    },
    payload_index_config={
        "indexed_fields": ["category", "tags", "created_at"],
        "text_fields": ["title", "source"]
    },
    shard_number=2,
    replication_factor=2,
    write_consistency_level="majority"
)

# Collection operations
collections = await client.collections.list(
    limit=100,
    include_stats=True
)

collection_info = await client.collections.get(
    name="technical_documents",
    include_aliases=True
)

# Update collection configuration
await client.collections.update(
    name="technical_documents",
    optimizers_config={
        "indexing_threshold": 10000,
        "max_segment_size": 200000
    }
)

# Collection aliases for zero-downtime updates
await client.collections.create_alias(
    collection_name="technical_documents_v2",
    alias_name="technical_documents"
)

# ==============================================================================
# 3. ADVANCED DOCUMENT INGESTION
# ==============================================================================

# Single document with validation
document = Document(
    id="doc_001",
    content="Comprehensive guide to transformer architectures and attention mechanisms...",
    metadata={
        "title": "Transformer Architecture Guide",
        "category": "technical",
        "tags": ["transformers", "attention", "deep-learning"],
        "source": "research_paper.pdf",
        "page_number": 1,
        "created_at": datetime.now().isoformat(),
        "author": "John Doe",
        "language": "en"
    }
)

# Validate before ingestion
validation_result = await client.validate.document(document)
if validation_result.is_valid:
    result = await client.ingest.document(
        collection="technical_documents",
        document=document,
        chunk_metadata_fields=["title", "source", "page_number"],
        wait_for_indexing=True
    )

# Batch ingestion with progress tracking and error handling
async def ingest_with_progress(documents: List[Document]):
    total = len(documents)
    processed = 0
    errors = []

    async for batch_result in client.ingest.documents(
        collection="technical_documents",
        documents=documents,
        batch_size=50,
        parallel_batches=4,  # Process 4 batches concurrently
        on_error="continue",  # "stop", "continue", or custom handler
        duplicate_strategy="update",  # "skip", "update", "error"
        show_progress=True
    ):
        processed += batch_result.success_count
        errors.extend(batch_result.errors)

        # Progress callback
        progress = (processed / total) * 100
        print(f"Progress: {progress:.1f}% ({processed}/{total})")

        # Handle errors
        if batch_result.errors:
            for error in batch_result.errors:
                print(f"Error in document {error.document_id}: {error.message}")

    return IngestionResult(
        total_count=total,
        success_count=processed,
        error_count=len(errors),
        errors=errors
    )

# File ingestion with automatic parsing
ingestion_config = {
    "parsers": {
        ".pdf": "pypdf",
        ".docx": "python-docx",
        ".txt": "text",
        ".md": "markdown",
        ".csv": "pandas"
    },
    "metadata_extractors": [
        {"field": "source", "extractor": "filename"},
        {"field": "created_at", "extractor": "file_mtime"},
        {"field": "file_size", "extractor": "file_size"},
        {"field": "file_type", "extractor": "mime_type"}
    ],
    "ocr_config": {
        "enable": True,
        "languages": ["eng", "fra"],
        "engine": "tesseract"
    }
}

await client.ingest.from_directory(
    collection="technical_documents",
    directory_path="./documents",
    glob_patterns=["**/*.pdf", "**/*.md", "**/*.txt"],
    recursive=True,
    config=ingestion_config,
    chunk_size=1000,
    exclude_patterns=["**/drafts/**", "**/.git/**"]
)

# URL ingestion with web scraping
await client.ingest.from_urls(
    collection="technical_documents",
    urls=[
        "https://example.com/blog/ml-guide",
        "https://example.com/docs/**"  # Crawl pattern
    ],
    max_depth=2,
    respect_robots_txt=True,
    user_agent="VectorSearchBot/1.0",
    rate_limit=1.0  # Requests per second
)

# ==============================================================================
# 4. COMPREHENSIVE SEARCH CAPABILITIES
# ==============================================================================

# Basic vector search with rich results
results = await client.search(
    collection="technical_documents",
    query="How do transformer models handle long-range dependencies?",
    top_k=10,
    score_threshold=0.7,
    include_metadata=True,
    include_vectors=False,
    timeout=10.0
)

# Advanced search with complex filters
results = await client.search(
    collection="technical_documents",
    query="attention mechanisms in NLP",
    top_k=20,
    filters={
        "$and": [
            {"category": {"$eq": "technical"}},
            {"tags": {"$in": ["transformers", "attention", "nlp"]}},
            {"created_at": {"$gte": "2024-01-01"}},
            {"$or": [
                {"author": {"$eq": "John Doe"}},
                {"source": {"$contains": "research"}}
            ]}
        ]
    },
    boost_fields={  # Field boosting
        "title": 2.0,
        "tags": 1.5
    },
    enable_reranking=True,
    rerank_config={
        "top_k": 5,
        "diversity": 0.3,
        "include_original_scores": True
    }
)

# Semantic search with query expansion
expanded_results = await client.search(
    collection="technical_documents",
    query="AI safety and alignment",
    search_type="semantic",
    query_expansion={
        "enable": True,
        "model": "gpt-3.5-turbo",
        "num_expansions": 3,
        "include_synonyms": True
    },
    context_window=3,  # Include surrounding chunks
    highlight_matches=True,
    highlight_config={
        "pre_tag": "<mark>",
        "post_tag": "</mark>",
        "fragment_size": 150
    }
)

# Multi-query search
queries = [
    "transformer architecture",
    "attention mechanism",
    "positional encoding"
]

multi_results = await client.search_multi(
    collection="technical_documents",
    queries=queries,
    fusion_method="reciprocal_rank",  # or "weighted", "max"
    query_weights=[0.5, 0.3, 0.2],
    top_k=10
)

# Conversational search with context
conversation_results = await client.conversational_search(
    collection="technical_documents",
    query="What are its limitations?",
    conversation_history=[
        {"role": "user", "content": "Tell me about BERT"},
        {"role": "assistant", "content": "BERT is a transformer-based model..."},
        {"role": "user", "content": "How does it handle context?"},
        {"role": "assistant", "content": "BERT uses bidirectional context..."}
    ],
    context_compression=True,
    max_history_tokens=1000
)

# ==============================================================================
# 5. RESULT PROCESSING AND EXPORT
# ==============================================================================

# Rich result object with methods
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score:.4f}")
    print(f"Rerank Score: {result.rerank_score:.4f}")
    print(f"Content: {result.content[:200]}...")
    print(f"Metadata: {result.metadata}")
    print(f"Highlights: {result.highlights}")
    print(f"Chunk Info: {result.chunk_info}")

    # Get surrounding context
    context = await result.get_context(window=2)
    print(f"Previous chunk: {context.previous}")
    print(f"Next chunk: {context.next}")

# Group and aggregate results
grouped = results.group_by("metadata.source")
document_scores = results.aggregate_by_document(method="max")  # or "mean", "sum"

# Export in various formats
await results.export(
    path="search_results.json",
    format="json",
    include_embeddings=False
)

await results.to_dataframe().to_csv("results.csv")  # Pandas integration

# Generate citation
citations = results.generate_citations(style="apa")  # or "mla", "chicago"

# ==============================================================================
# 6. ANALYTICS AND MONITORING
# ==============================================================================

# Collection analytics
stats = await client.analytics.collection_stats(
    collection="technical_documents",
    include_distribution=True
)

print(f"Total vectors: {stats.vector_count}")
print(f"Total documents: {stats.document_count}")
print(f"Average chunks per document: {stats.avg_chunks_per_doc:.2f}")
print(f"Storage size: {stats.storage_size_mb:.2f} MB")
print(f"Index size: {stats.index_size_mb:.2f} MB")
print(f"Metadata distribution: {stats.metadata_distribution}")

# Search analytics with time series
search_metrics = await client.analytics.search_metrics(
    collection="technical_documents",
    time_range="7d",
    granularity="hour",
    group_by=["query_type", "user_id"]
)

# Performance monitoring
perf = await client.performance.get_metrics()
print(f"Average search latency: {perf.avg_search_latency_ms:.2f}ms")
print(f"P95 search latency: {perf.p95_search_latency_ms:.2f}ms")
print(f"Indexing throughput: {perf.indexing_throughput_docs_per_sec:.2f} docs/sec")
print(f"Query throughput: {perf.queries_per_second:.2f} QPS")

# Cost estimation
cost_estimate = await client.analytics.estimate_cost(
    storage_gb=stats.storage_size_mb / 1024,
    monthly_queries=1_000_000,
    provider="qdrant"
)

# ==============================================================================
# 7. ADVANCED FEATURES
# ==============================================================================

# A/B Testing for retrieval strategies
ab_test = await client.experiments.create_ab_test(
    name="retrieval_strategy_test",
    variants={
        "control": VectorRetriever(top_k=10),
        "treatment": HybridRetriever(top_k=15, vector_weight=0.7)
    },
    traffic_allocation={"control": 0.5, "treatment": 0.5},
    metrics=["click_through_rate", "avg_result_position", "user_satisfaction"],
    duration_days=14,
    minimum_sample_size=1000
)

# Get A/B test results
ab_results = await client.experiments.get_results("retrieval_strategy_test")
if ab_results.is_significant:
    print(f"Winner: {ab_results.winner}")
    print(f"Lift: {ab_results.lift_percentage:.2f}%")

# Recommendation system
recommendations = await client.recommend(
    collection="technical_documents",
    positive_examples=["doc_001", "doc_005", "doc_007"],
    negative_examples=["doc_010", "doc_015"],
    user_profile={
        "interests": ["machine-learning", "nlp"],
        "expertise_level": "intermediate"
    },
    top_k=10,
    diversity_weight=0.3
)

# Clustering with multiple algorithms
clusters = await client.cluster(
    collection="technical_documents",
    algorithm="hdbscan",  # or "kmeans", "dbscan", "agglomerative"
    min_cluster_size=5,
    min_samples=3,
    cluster_selection_epsilon=0.5,
    return_outliers=True,
    return_cluster_centers=True
)

# Duplicate detection
duplicates = await client.find_duplicates(
    collection="technical_documents",
    similarity_threshold=0.95,
    algorithm="minhash",  # or "simhash", "exact"
    group_similar=True
)

# ==============================================================================
# 8. PIPELINE AND WORKFLOW SYSTEM
# ==============================================================================

# Create reusable pipeline
pipeline = (
    client.pipelines.create("document_processing")
    .add_stage("validate",
        lambda doc: client.validate.document(doc)
    )
    .add_stage("chunk",
        RecursiveTextSplitter(chunk_size=500)
    )
    .add_stage("embed",
        HuggingFaceEmbedder(model_id="all-MiniLM-L6-v2"),
        parallel=True,
        batch_size=32
    )
    .add_stage("enrich",
        lambda doc: enrich_with_metadata(doc),
        timeout=30
    )
    .add_stage("store",
        client.ingest.document,
        retry_on_failure=True
    )
    .add_stage("index",
        lambda: client.collections.optimize("technical_documents")
    )
    .with_error_handler(custom_error_handler)
    .with_progress_callback(progress_callback)
    .build()
)

# Execute pipeline
pipeline_result = await pipeline.execute(
    input_data=documents,
    collection="technical_documents",
    dry_run=False
)

# Schedule pipeline
schedule = await pipeline.schedule(
    name="daily_ingestion",
    cron="0 2 * * *",  # 2 AM daily
    input_source={
        "type": "s3",
        "bucket": "my-bucket",
        "prefix": "daily-docs/",
        "delete_after_processing": True
    },
    notifications={
        "on_success": "slack://channel/general",
        "on_failure": "email://admin@example.com"
    }
)

# ==============================================================================
# 9. BACKUP, MIGRATION, AND DISASTER RECOVERY
# ==============================================================================

# Create backup with versioning
backup = await client.backup.create(
    collections=["technical_documents"],
    destination="s3://my-bucket/backups/",
    format="parquet",  # or "json", "arrow"
    compression="snappy",
    include_vectors=True,
    include_metadata=True,
    encryption_key="${BACKUP_ENCRYPTION_KEY}"
)

# Incremental backup
incremental_backup = await client.backup.create_incremental(
    collections=["technical_documents"],
    since=datetime(2024, 1, 1),
    destination="s3://my-bucket/incremental/"
)

# Restore from backup
await client.backup.restore(
    source=backup.backup_id,
    collections=["technical_documents"],
    strategy="merge",  # or "replace", "append"
    validate_before_restore=True
)

# Migration between databases
migration = await client.migrate.create(
    source_db=QdrantDB(url="http://old-instance:6333"),
    target_db=PineconeDB(api_key="..."),
    collections=["technical_documents"],
    batch_size=1000,
    preserve_ids=True,
    verify_migration=True
)

# ==============================================================================
# 10. SECURITY AND COMPLIANCE
# ==============================================================================

# Configure security
client.security.configure(
    encryption_at_rest=True,
    encryption_in_transit=True,
    audit_logging=True,
    ip_whitelist=["10.0.0.0/8"],
    rate_limiting={
        "requests_per_minute": 1000,
        "requests_per_hour": 50000
    }
)

# Data privacy and GDPR compliance
# Anonymize PII in documents
await client.privacy.anonymize_collection(
    collection="technical_documents",
    pii_detection_model="presidio",
    fields_to_check=["content", "metadata.author"],
    anonymization_method="redact"  # or "hash", "replace"
)

# Right to be forgotten
await client.privacy.delete_user_data(
    user_id="user_123",
    collections=["technical_documents"],
    cascade=True
)

# Audit trail
audit_logs = await client.security.get_audit_logs(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now(),
    event_types=["search", "ingest", "delete"],
    user_id="user_123"
)

# ==============================================================================
# 11. HEALTH CHECKS AND DIAGNOSTICS
# ==============================================================================

# Comprehensive health check
health = await client.health_check(detailed=True)
print(f"Overall status: {health.status}")
print(f"Database: {health.components.database.status}")
print(f"Embedder: {health.components.embedder.status}")
print(f"Cache: {health.components.cache.status}")
print(f"Latency check: {health.latency_ms}ms")

# Diagnostics
diagnostics = await client.diagnostics.run()
if diagnostics.issues:
    for issue in diagnostics.issues:
        print(f"Issue: {issue.severity} - {issue.description}")
        print(f"Recommendation: {issue.recommendation}")

# Performance profiling
profile = await client.diagnostics.profile_operation(
    operation="search",
    collection="technical_documents",
    sample_size=100
)

# ==============================================================================
# 12. INTEGRATIONS
# ==============================================================================

# LangChain integration
from langchain.vectorstores import VectorSearchLibraryStore

langchain_store = VectorSearchLibraryStore(
    client=client,
    collection_name="technical_documents"
)

# LlamaIndex integration
from llama_index.vector_stores import VectorSearchLibraryVectorStore

llama_store = VectorSearchLibraryVectorStore(
    client=client,
    collection_name="technical_documents"
)

# Haystack integration
from haystack.document_stores import VectorSearchLibraryDocumentStore

haystack_store = VectorSearchLibraryDocumentStore(
    client=client,
    collection_name="technical_documents"
)

# FastAPI integration
from fastapi import FastAPI
from vectorkit.integrations.fastapi import VectorSearchRouter

app = FastAPI()
app.include_router(
    VectorSearchRouter(client=client),
    prefix="/api/v1/search"
)

# ==============================================================================
# 13. CLI INTERFACE
# ==============================================================================

# CLI usage examples (when installed)
"""
# Initialize configuration
vector-search init --database qdrant --embedder openai

# Create collection
vector-search collection create technical_documents --dimension 384

# Ingest documents
vector-search ingest --collection technical_documents --path ./docs/

# Search
vector-search search --collection technical_documents --query "machine learning"

# Export results
vector-search export --collection technical_documents --format parquet --output ./export/
"""

# ==============================================================================
# 14. ERROR HANDLING PATTERNS
# ==============================================================================

# Comprehensive error handling
try:
    results = await client.search(
        collection="technical_documents",
        query="test query",
        top_k=10
    )
except CollectionNotFoundError as e:
    # Handle missing collection
    logger.error(f"Collection not found: {e.collection_name}")
    await client.collections.create(e.collection_name)
    # Retry operation
except EmbeddingError as e:
    # Handle embedding failures
    logger.error(f"Embedding failed: {e}")
    # Fallback to keyword search
    results = await client.search(
        collection="technical_documents",
        query="test query",
        search_type="keyword"
    )
except RateLimitError as e:
    # Handle rate limiting
    logger.warning(f"Rate limited. Retry after: {e.retry_after}s")
    await asyncio.sleep(e.retry_after)
    # Retry with exponential backoff
except DatabaseConnectionError as e:
    # Handle connection issues
    logger.error(f"Database connection failed: {e}")
    # Try fallback database
    client.switch_to_fallback_database()
except VectorSearchError as e:
    # General error handling
    logger.error(f"Operation failed: {e}")
    # Send alert
    await send_alert(f"Vector search error: {e}")
finally:
    # Cleanup
    await client.close()

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    # Initialize client
    client = VectorSearchClient.from_config("production.yaml")

    # Create or get collection
    collection = await client.collections.get_or_create(
        name="technical_documents",
        vector_config={"size": 384, "distance": "cosine"}
    )

    # Ingest documents with progress tracking
    documents = load_documents("./data/")
    async for progress in client.ingest.documents_with_progress(
        collection="technical_documents",
        documents=documents,
        batch_size=50
    ):
        print(f"Progress: {progress.percentage:.1f}%")

    # Perform search
    results = await client.search(
        collection="technical_documents",
        query="explain transformer attention mechanisms",
        top_k=5,
        enable_reranking=True
    )

    # Process results
    for result in results:
        print(f"Score: {result.score:.3f} - {result.content[:100]}...")

    # Get analytics
    stats = await client.analytics.collection_stats("technical_documents")
    print(f"Total documents: {stats.document_count}")

    # Cleanup
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration File Example (production.yaml)

```yaml
vector_search:
  database:
    type: qdrant
    config:
      url: ${QDRANT_URL}
      api_key: ${QDRANT_API_KEY}
      timeout: 30
      grpc_port: 6334
      prefer_grpc: true
      retry_config:
        max_retries: 3
        backoff_factor: 2.0

  embedder:
    type: huggingface
    config:
      model_id: sentence-transformers/all-MiniLM-L6-v2
      device: auto
      batch_size: 32
      normalize_embeddings: true
      cache_folder: ./embeddings_cache

  chunker:
    type: recursive
    config:
      chunk_size: 1000
      chunk_overlap: 200
      separators: ["\n\n", "\n", ". ", " "]

  retriever:
    type: hybrid
    config:
      vector_weight: 0.7
      keyword_weight: 0.3
      top_k: 20

  reranker:
    type: cross_encoder
    config:
      model_id: cross-encoder/ms-marco-MiniLM-L-12-v2
      batch_size: 16
      top_k: 10

  cache:
    type: redis
    config:
      host: ${REDIS_HOST}
      port: 6379
      ttl: 3600
      max_size_mb: 1024

  monitoring:
    enable_metrics: true
    enable_tracing: true
    enable_audit_log: true
    metrics_port: 9090

  security:
    encryption_at_rest: true
    encryption_in_transit: true
    ip_whitelist:
      - 10.0.0.0/8
      - 172.16.0.0/12
```
