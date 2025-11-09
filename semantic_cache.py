import time
import uuid
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import redis
import requests
from sentence_transformers import SentenceTransformer


@dataclass
class CacheResult:
    response: str
    is_cached: bool
    similarity: float
    latency_seconds: float
    key: Optional[str]


class SemanticCache:
    """Semantic cache backed by Redis with a vector index (RediSearch).

    - Stores each query as a Redis Hash with fields: text, response, embedding (FLOAT32 bytes)
    - Vector index: HNSW on field `embedding` with cosine distance
    - Provides `handle_query` which returns a CacheResult with timing and cache flag
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        index_name: str = "idx:semantic_cache",
        key_prefix: str = "sc:",
        model_name: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3",
        similarity_threshold: float = 0.85,
    ):
        self.redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=False)
        # use bytes; we'll decode selected fields manually
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.vector_field = "embedding"
        self.text_field = "text"
        self.response_field = "response"
        self.dim = 384  # all-MiniLM-L6-v2
        self.distance_metric = "COSINE"
        self.similarity_threshold = similarity_threshold
        self.ollama_model = ollama_model

        # Load embedding model once
        self.embedder = SentenceTransformer(model_name)

        # Ensure the vector index exists
        self._ensure_index()

    # ------------------------ Redis Index Setup ------------------------
    def _ensure_index(self):
        try:
            self.redis.ft(self.index_name).info()
            return
        except Exception:
            pass

        from redis.commands.search.field import TextField, VectorField
        try:
            # redis-py 4.x
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType  # type: ignore
        except ModuleNotFoundError:  # redis-py 5.x uses snake_case module name
            from redis.commands.search.index_definition import IndexDefinition, IndexType  # type: ignore

        schema = (
            TextField(self.text_field),
            TextField(self.response_field),
            VectorField(
                self.vector_field,
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": self.distance_metric,
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )

        definition = IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)
        try:
            self.redis.ft(self.index_name).create_index(schema, definition=definition)
        except Exception as e:
            msg = str(e).lower()
            if "unknown command" in msg and "ft.create" in msg:
                raise RuntimeError(
                    "Redis server does not support RediSearch (vector indices). Start Redis Stack: "
                    "docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest "
                    "or brew install redis-stack-server && brew services start redis-stack-server"
                ) from e
            raise

    # --------------------------- Utilities ----------------------------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], normalize_embeddings=True)[0]
        # ensure float32 in bytes for Redis
        return np.asarray(vec, dtype=np.float32)

    def _key(self) -> str:
        return f"{self.key_prefix}{uuid.uuid4().hex}"

    # --------------------------- LLM Call ------------------------------
    def _call_ollama(self, prompt: str, timeout: int = 120) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Standard Ollama returns {response: "...", done: true, ...}
        return data.get("response", "")

    # --------------------------- Public API ---------------------------
    def handle_query(self, query: str, k: int = 1) -> CacheResult:
        start = time.time()
        embedding = self._embed(query)

        # Search nearest neighbor in Redis
        try:
            from redis.commands.search.query import Query

            base_q = f"*=>[KNN {k} @{self.vector_field} $vec_param AS score]"
            params = {"vec_param": embedding.tobytes()}
            q = (
                Query(base_q)
                .return_fields(self.text_field, self.response_field, "score")
                .sort_by("score")
                .paging(0, k)
                .dialect(2)
            )
            results = self.redis.ft(self.index_name).search(q, query_params=params)
        except Exception as e:
            # If index is missing or RediSearch not available, behave as cache miss path
            results = None

        best_similarity = 0.0
        best_doc: Optional[Dict[str, bytes]] = None

        if results and getattr(results, "docs", None):
            doc = results.docs[0]
            # RediSearch returns cosine distance as `score`; similarity = 1 - distance (when normalized)
            try:
                dist = float(doc.score)
                best_similarity = max(0.0, 1.0 - dist)
            except Exception:
                best_similarity = 0.0
            best_doc = {
                self.text_field: getattr(doc, self.text_field, b""),
                self.response_field: getattr(doc, self.response_field, b""),
            }

        if best_doc is not None and best_similarity >= self.similarity_threshold:
            # Cache hit
            response_bytes = best_doc[self.response_field]
            try:
                response = response_bytes.decode("utf-8") if isinstance(response_bytes, (bytes, bytearray)) else str(response_bytes)
            except Exception:
                response = str(response_bytes)
            latency = time.time() - start
            return CacheResult(response=response, is_cached=True, similarity=best_similarity, latency_seconds=latency, key=None)

        # Cache miss -> call LLM
        llm_start = time.time()
        response = self._call_ollama(query)
        latency = time.time() - llm_start

        # Store in Redis for future hits
        key = self._key()
        mapping = {
            self.text_field: query.encode("utf-8"),
            self.response_field: response.encode("utf-8"),
            self.vector_field: embedding.tobytes(),
            b"ts": str(time.time()).encode("utf-8"),
        }
        self.redis.hset(key, mapping=mapping)

        # Return result with miss flag
        total_latency = time.time() - start
        return CacheResult(response=response, is_cached=False, similarity=best_similarity, latency_seconds=total_latency, key=key)


def run_demo():
    cache = SemanticCache()

    queries: List[str] = [
        "What is the capital of France?",
        "Explain the concept of overfitting in machine learning.",
        "How do I reverse a list in Python?",
        "What is the capital city of France?",  # paraphrase
        "Describe overfitting in ML and how to prevent it.",  # paraphrase
        "How can I invert a Python list?",  # paraphrase
        "Write a haiku about the ocean.",
        "Summarize the benefits of unit testing.",
        "What are HTTP status codes?",
        "What's 2 + 2?",
        "Tell me about HTTP response codes.",  # paraphrase
        "What is the capital of France?",  # exact duplicate
    ]

    print("=== Semantic Cache Demo ===")
    hits = 0
    misses = 0
    cached_times: List[float] = []
    miss_times: List[float] = []

    for i, q in enumerate(queries, 1):
        res = cache.handle_query(q)
        src = "CACHE" if res.is_cached else "OLLAMA"
        if res.is_cached:
            hits += 1
            cached_times.append(res.latency_seconds)
        else:
            misses += 1
            miss_times.append(res.latency_seconds)
        print(f"{i:02d}. [{src}] sim={res.similarity:.3f} time={res.latency_seconds:.3f}s\n   Q: {q}\n   A: {res.response[:100].strip()}...\n")

    total = hits + misses
    hit_rate = (hits / total) * 100 if total else 0.0
    avg_cached = float(np.mean(cached_times)) if cached_times else 0.0
    avg_miss = float(np.mean(miss_times)) if miss_times else 0.0
    speedup = (avg_miss / avg_cached) if (avg_cached and avg_miss) else 0.0

    print("=== Metrics ===")
    print(f"Queries: {total}, Hits: {hits}, Misses: {misses}, Hit Rate: {hit_rate:.1f}%")
    print(f"Avg cached time: {avg_cached:.3f}s, Avg non-cached time: {avg_miss:.3f}s")
    print(f"Speedup (cached vs non-cached): {speedup:.1f}x")


if __name__ == "__main__":
    run_demo()


