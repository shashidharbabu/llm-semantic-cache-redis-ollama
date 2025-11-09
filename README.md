# LLM Semantic Cache with Redis and Ollama

A high-performance semantic caching system that stores Ollama LLM responses in Redis using vector embeddings. This project implements two Redis-based caching solutions:

1. **View Counter with Redis** - Efficient post view tracking with periodic database sync
2. **Semantic Caching** - Vector-based semantic similarity search for LLM query caching

## Features

### Task 1: Redis-Backed View Counter
- Increments view counts in Redis using `INCR` command
- Syncs to SQLite database every 10 views
- Provides significant performance improvement over direct database writes

### Task 2: Semantic Caching System
- **Vector Embeddings**: Uses SentenceTransformers (`all-MiniLM-L6-v2`) to generate query embeddings
- **Redis Vector Index**: HNSW index with cosine similarity for fast semantic search
- **Similarity Threshold**: Configurable threshold (default: 0.85) for cache hits
- **Ollama Integration**: Automatically calls Ollama when no similar query is found
- **Performance Tracking**: Measures cache hit rate, response times, and speedup metrics
- **Cache Hit Rate**: Tracks percentage of queries served from cache vs. Ollama

## Requirements

- Python 3.9+
- Redis Stack (with RediSearch module for vector search)
- Ollama (running locally on port 11434)
- Required Python packages:
  ```
  redis
  sentence-transformers
  numpy
  requests
  ```

## Installation

1. **Install Redis Stack**:
   ```bash
   # Using Docker
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
   
   # Or using Homebrew (macOS)
   brew install redis-stack-server
   brew services start redis-stack-server
   ```

2. **Install Python dependencies**:
   ```bash
   pip install redis sentence-transformers numpy requests
   ```

3. **Install and run Ollama**:
   - Download from [ollama.ai](https://ollama.ai)
   - Pull the model: `ollama pull llama3`

## Usage

### Task 1: View Counter Demo

```bash
python3 simulate.py
```

This demonstrates:
- Incremental view counting in Redis
- Automatic database sync every 10 views
- Performance comparison between Redis-backed and direct database writes

### Task 2: Semantic Cache Demo

```bash
python3 semantic_demo.py
```

This runs 12 diverse queries including:
- Exact duplicate queries
- Paraphrased questions
- Completely new queries

The demo outputs:
- Cache hits/misses with similarity scores
- Response times for each query
- Overall metrics: hit rate, average response times, speedup factor

## Project Structure

```
.
├── demo.py                 # DatabaseWithCache class with view counter
├── semantic_cache.py       # SemanticCache class with vector search
├── semantic_demo.py        # Demo script for semantic caching
├── simulate.py            # Demo script for view counter
├── demo.db                # SQLite database (auto-generated)
└── README.md              # This file
```

## Performance Metrics

The semantic cache typically achieves:
- **Cache Hit Rate**: 16-30% (varies with query diversity)
- **Speedup**: 50-100x faster for cached responses vs. Ollama calls
- **Average Cached Response Time**: < 1 second
- **Average Non-Cached Response Time**: 30-80 seconds (depends on Ollama)

## Key Implementation Details

### Semantic Cache
- Uses HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search
- Embeddings are normalized for cosine similarity calculation
- Cache entries stored as Redis Hashes with fields: `text`, `response`, `embedding`
- Similarity score calculated as `1 - cosine_distance` (normalized embeddings)

### View Counter
- Redis key format: `views:post:{post_id}`
- Database sync occurs when `view_count % 10 == 0`
- Maintains consistency between Redis cache and SQLite database
