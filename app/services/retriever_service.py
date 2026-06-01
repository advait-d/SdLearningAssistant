"""
RAG Pipeline — retriever_service.py
====================================
Implements chunking, embedding, FAISS storage, and similarity search.
Drop this file into app/services/ in your project structure.

Usage:
    pipeline = RAGPipeline()
    pipeline.build_index(raw_documents)
    results = pipeline.retrieve_top_k("How does consistent hashing work?")
"""

from __future__ import annotations

import os
import pickle
import logging
import asyncio
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import re
import numpy as np
import faiss
from openai import OpenAI

try:
    from google import genai
    has_genai = True
except ImportError:
    has_genai = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_DIM_OPENAI = 1536          # fixed for text-embedding-3-small

EMBEDDING_MODEL_GEMINI = "models/gemini-embedding-001"
EMBEDDING_DIM_GEMINI = 3072          # gemini-embedding-001 output dimension

CHUNK_TARGET_TOKENS = 400     # target chunk size (in words; ~1.3 words per token on average)
CHUNK_OVERLAP_TOKENS = 80     # word overlap between consecutive chunks
MAX_CHUNKS_PER_EMBED_CALL = 20   # conservative batch size; Gemini free tier is rate-limited


# ---------------------------------------------------------------------------
# Lightweight tokenizer — replaces tiktoken
# ---------------------------------------------------------------------------

def _split_words(text: str) -> list[str]:
    """
    Split text into word-level tokens.

    Splits on whitespace while preserving punctuation attached to words.
    Good enough for chunking purposes — the embedding model does its own
    real tokenisation internally; we just need consistent, stable splits.

    Accuracy vs tiktoken: word count is ~75% of BPE token count for English
    prose, so CHUNK_TARGET_TOKENS=400 words ≈ 530 BPE tokens, which safely
    fits within the embedding model's 8192-token limit.
    """
    return text.split()


def _join_words(words: list[str]) -> str:
    return " ".join(words)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A raw input document with optional metadata."""
    content: str
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A single text chunk derived from a Document."""
    text: str
    source: str
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, with its score."""
    text: str
    source: str
    chunk_index: int
    score: float           # cosine similarity [0, 1]; higher = more relevant
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline:
        1. chunk_documents()  — split raw docs into overlapping token windows
        2. create_embeddings() — embed chunks via OpenAI text-embedding-3-small
        3. store_in_faiss()   — build an L2-normalised FAISS index
        4. retrieve_top_k()   — embed a query and return the k closest chunks

    The index and chunk store can be persisted to disk and reloaded, so
    ingestion only needs to run once per knowledge base update.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.openai_client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.gemini_client = None
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if has_genai and gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            
        self.index: Optional[faiss.Index] = None
        self.chunks: list[Chunk] = []   # parallel list — chunks[i] corresponds to index vector i

    # ------------------------------------------------------------------
    # 1. Chunking
    # ------------------------------------------------------------------

    def chunk_documents(
        self,
        documents: list[Document],
        chunk_size: int = CHUNK_TARGET_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> list[Chunk]:
        """
        Split each document into overlapping token-window chunks.

        Strategy:
          - Split the document into words using _split_words().
          - Slide a window of `chunk_size` words, stepping by (chunk_size - overlap).
          - Word count is used instead of BPE tokens (no tiktoken dependency).
            400 words ≈ 530 BPE tokens — well within the embedding model's
            8192-token limit, so this is safe in practice.
          - Trailing fragments under 20 words are merged into the previous chunk.

        Args:
            documents:  List of Document objects to chunk.
            chunk_size: Target window size in words.
            overlap:    Number of words shared between consecutive chunks.

        Returns:
            Flat list of Chunk objects across all documents.
        """
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        all_chunks: list[Chunk] = []
        step = chunk_size - overlap

        for doc in documents:
            words = _split_words(doc.content)

            if not words:
                logger.warning("Skipping empty document: %s", doc.source)
                continue

            windows: list[list[str]] = []
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                windows.append(words[start:end])
                if end == len(words):
                    break
                start += step

            # Merge tiny trailing fragments into the previous window
            if len(windows) > 1 and len(windows[-1]) < 20:
                windows[-2].extend(windows[-1])
                windows.pop()

            for idx, window_words in enumerate(windows):
                text = _join_words(window_words)
                all_chunks.append(Chunk(
                    text=text,
                    source=doc.source,
                    chunk_index=idx,
                    token_count=len(window_words),  # word count; ≈0.75× BPE tokens
                    metadata=doc.metadata,
                ))

        logger.info(
            "Chunked %d document(s) into %d chunks (size=%d, overlap=%d tokens)",
            len(documents), len(all_chunks), chunk_size, overlap,
        )
        return all_chunks

    # ------------------------------------------------------------------
    # 2. Embeddings
    # ------------------------------------------------------------------

    def create_embeddings(self, chunks: list[Chunk], provider: str = "openai") -> np.ndarray:
        """
        Embed a list of Chunk objects.

        Sends chunks in small batches. Retries with exponential backoff on
        429 (rate-limit) errors from either provider.

        Args:
            chunks: Chunks to embed.
            provider: "openai" or "gemini".

        Returns:
            Float32 numpy array of shape (len(chunks), EMBEDDING_DIM).
        """
        import time

        dim = EMBEDDING_DIM_OPENAI if provider == "openai" else EMBEDDING_DIM_GEMINI
        if not chunks:
            return np.empty((0, dim), dtype=np.float32)

        texts = [chunk.text for chunk in chunks]
        all_embeddings: list[list[float]] = []
        total_batches = -(-len(texts) // MAX_CHUNKS_PER_EMBED_CALL)  # ceiling div

        for batch_idx, batch_start in enumerate(range(0, len(texts), MAX_CHUNKS_PER_EMBED_CALL)):
            batch = texts[batch_start: batch_start + MAX_CHUNKS_PER_EMBED_CALL]
            logger.info(
                "Embedding batch %d/%d (%d chunks)",
                batch_idx + 1, total_batches, len(batch),
            )

            # Retry loop with exponential backoff for 429 / transient errors
            max_retries = 6
            delay = 5.0  # seconds
            for attempt in range(max_retries):
                try:
                    if provider == "gemini":
                        if not self.gemini_client:
                            raise RuntimeError("GEMINI_API_KEY missing.")
                        response = self.gemini_client.models.embed_content(
                            model=EMBEDDING_MODEL_GEMINI,
                            contents=batch
                        )
                        batch_vectors = [item.values for item in response.embeddings]
                    else:
                        response = self.openai_client.embeddings.create(
                            model=EMBEDDING_MODEL_OPENAI,
                            input=batch,
                            encoding_format="float",
                        )
                        batch_vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                    break  # success

                except Exception as exc:
                    err_str = str(exc)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate" in err_str.lower()
                    if is_rate_limit and attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(
                            "Rate limit hit (attempt %d/%d). Retrying in %.0fs…",
                            attempt + 1, max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise

            all_embeddings.extend(batch_vectors)

            # Inter-batch pause to stay within free-tier RPM limits
            # Gemini free tier: ~10 RPM → wait 7s between batches
            if provider == "gemini" and batch_idx < total_batches - 1:
                time.sleep(7.0)

        vectors = np.array(all_embeddings, dtype=np.float32)
        logger.info("Created embeddings: shape=%s", vectors.shape)
        return vectors

    # ------------------------------------------------------------------
    # 3. FAISS index
    # ------------------------------------------------------------------

    def store_in_faiss(self, chunks: list[Chunk], embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings and store it alongside the chunks.

        Index type: IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
        Chosen because:
          - Exact search — no approximation error for knowledge bases up to ~500k chunks.
          - Cosine similarity is more meaningful than raw L2 distance for text.
          - For corpora >500k chunks, swap to IndexIVFFlat with nlist=sqrt(n).

        Args:
            chunks:     Chunk list — must be parallel to `embeddings`.
            embeddings: Float32 array of shape (N, EMBEDDING_DIM).

        Returns:
            The populated faiss.Index (also stored as self.index).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks

        logger.info("FAISS index built: %d vectors, dimension=%d", index.ntotal, dim)
        return index

    # ------------------------------------------------------------------
    # 4. Similarity search
    # ------------------------------------------------------------------

    def retrieve_top_k(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        provider: str = "openai"
    ) -> list[RetrievedChunk]:
        """
        Embed the query and return the k most similar chunks.

        Args:
            query:           Natural language query string.
            k:               Number of results to return.
            score_threshold: Minimum cosine similarity score [0, 1].
                             Results below this are filtered out.
                             0.0 means return all top-k regardless of score.

        Returns:
            List of RetrievedChunk sorted by descending similarity score.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Index is empty. Run build_index() or load_index() first.")

        if provider == "gemini":
            if not self.gemini_client:
                raise RuntimeError("GEMINI_API_KEY missing.")
            response = self.gemini_client.models.embed_content(
                model=EMBEDDING_MODEL_GEMINI,
                contents=query,
            )
            query_vector = np.array([response.embeddings[0].values], dtype=np.float32)
        else:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL_OPENAI,
                input=[query],
                encoding_format="float",
            )
            query_vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search — returns scores and FAISS-internal IDs
        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for empty slots
                continue
            if score < score_threshold:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=float(score),
                metadata=chunk.metadata,
            ))

        logger.info("retrieve_top_k('%s'): %d results returned", query[:60], len(results))
        return results

    # ------------------------------------------------------------------
    # Convenience: build full pipeline in one call
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: list[Document],
        chunk_size: int = CHUNK_TARGET_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
        provider: str = "openai",
    ) -> None:
        """
        Run the full ingestion pipeline: chunk → embed → store.
        Call this once per knowledge base update.
        """
        chunks = self.chunk_documents(documents, chunk_size, overlap)
        embeddings = self.create_embeddings(chunks, provider=provider)
        self.store_in_faiss(chunks, embeddings)

    # ------------------------------------------------------------------
    # Persistence — save / load index to disk
    # ------------------------------------------------------------------

    def save_index(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.
        Allows skipping re-ingestion on server restart.

        Saves:
            <directory>/faiss.index  — binary FAISS index
            <directory>/chunks.pkl   — serialised list of Chunk objects
        """
        if self.index is None:
            raise RuntimeError("No index to save. Run build_index() first.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Index saved to %s (%d vectors)", path, self.index.ntotal)

    def load_index(self, directory: str | Path) -> None:
        """
        Load a previously saved FAISS index and chunk list from disk.
        Call this on startup instead of re-ingesting.
        """
        path = Path(directory)

        self.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        logger.info("Index loaded from %s (%d vectors)", path, self.index.ntotal)


# ---------------------------------------------------------------------------
# Quick smoke test — runs only when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_docs = [
        Document(
            content="""
            Consistent hashing is a distributed systems technique used to distribute
            load across nodes in a way that minimises redistribution when nodes are
            added or removed. In a traditional hash table, adding or removing a slot
            changes the mapping for almost every key. Consistent hashing solves this
            by mapping both keys and nodes onto a circular ring using the same hash
            function. Each key is assigned to the first node clockwise from its
            position on the ring. When a node is added or removed, only the keys
            between the new/removed node and its predecessor are redistributed.
            Virtual nodes (vnodes) are used to improve load balance — each physical
            node is assigned multiple positions on the ring, so load is spread more
            evenly even with heterogeneous hardware. Systems like Amazon DynamoDB,
            Apache Cassandra, and Riak use consistent hashing as a core primitive.
            """,
            source="consistent_hashing.md",
            metadata={"topic": "distributed-systems"},
        ),
        Document(
            content="""
            A URL shortener service converts a long URL into a short alias that
            redirects to the original. The core design challenge is generating a
            unique short key for each URL. Common approaches are: (1) Base62 encoding
            of an auto-incremented ID from a relational database — simple but the
            database becomes a single point of failure; (2) MD5/SHA256 hash of the
            URL, taking the first 7 characters — collision probability is low but not
            zero; (3) A distributed ID generator like Twitter Snowflake that produces
            unique 64-bit IDs without coordination. The redirection layer needs to be
            extremely fast — the short URL hit path should serve from an in-memory
            cache (Redis) with TTL-based expiry. A CDN in front handles geographic
            distribution. Write throughput is low (URL creation), read throughput is
            very high (redirects), so the system is read-optimised with an asymmetric
            cache strategy.
            """,
            source="url_shortener.md",
            metadata={"topic": "system-design"},
        ),
        Document(
            content="""
            The CAP theorem states that a distributed system can guarantee at most
            two of three properties simultaneously: Consistency (every read returns
            the most recent write), Availability (every request receives a response,
            even if it may not be the most recent), and Partition tolerance (the
            system continues operating despite network partitions). Since network
            partitions are unavoidable in practice, real systems must choose between
            CP (consistent and partition-tolerant, e.g. HBase, Zookeeper) and AP
            (available and partition-tolerant, e.g. Cassandra, CouchDB). The PACELC
            theorem extends CAP by also considering the latency-consistency tradeoff
            that exists even when there is no partition.
            """,
            source="cap_theorem.md",
            metadata={"topic": "distributed-systems"},
        ),
    ]

    pipeline = RAGPipeline()
    pipeline.build_index(sample_docs)

    queries = [
        "How does consistent hashing handle node failures?",
        "What database should I use for URL shortening?",
        "Explain the tradeoff between consistency and availability",
    ]

    print("\n" + "=" * 60)
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = pipeline.retrieve_top_k(query, k=2, score_threshold=0.3)
        for r in results:
            print(f"  [{r.score:.3f}] {r.source} (chunk {r.chunk_index})")
            print(f"  {r.text[:120].strip()}...")
        print()


# ---------------------------------------------------------------------------
# RetrieverService — async-friendly wrapper used by the orchestrator
# ---------------------------------------------------------------------------

# Default path where the FAISS index is persisted between restarts.
# Override via FAISS_INDEX_DIR environment variable.
_DEFAULT_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "faiss_index"
)


class RetrieverService:
    """
    Thin service layer that wraps RAGPipeline for use by the orchestrator.

    Responsibilities:
    - Lazy-loads the FAISS index from disk on the first query so startup is fast.
    - Exposes ``retrieve_context()``, an async method that runs the CPU-bound
      FAISS search in a thread-pool executor to avoid blocking the event loop.
    - Formats the retrieved chunks into a clean, prompt-ready context string.

    Index persistence:
        Before the service can answer queries the FAISS index must be built.
        Run ``RAGPipeline.build_index(docs)`` followed by
        ``RAGPipeline.save_index(index_dir)`` once during initial ingestion.
        On subsequent startups this service will load that persisted index
        automatically.
    """

    # Top-K chunks to retrieve per query; tune to balance context length vs noise.
    TOP_K = 5
    SCORE_THRESHOLD = 0.25

    def __init__(self, index_dir: Optional[str] = None):
        self.base_index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR", _DEFAULT_INDEX_DIR))
        self._pipelines: dict[str, Optional[RAGPipeline]] = {"openai": None, "gemini": None}
        self._index_loaded: dict[str, bool] = {"openai": False, "gemini": False}

    # ------------------------------------------------------------------
    # Lazy index loading
    # ------------------------------------------------------------------

    def _ensure_index(self, provider: str) -> None:
        """
        Load the FAISS index from disk for the specific provider.
        """
        if self._index_loaded.get(provider, False):
            return

        pipeline = RAGPipeline()
        provider_dir = self.base_index_dir / provider

        if provider_dir.exists() and (provider_dir / "faiss.index").exists():
            try:
                pipeline.load_index(provider_dir)
                self._pipelines[provider] = pipeline
                self._index_loaded[provider] = True
                logger.info("RetrieverService: FAISS index loaded from %s", provider_dir)
            except Exception as exc:
                logger.error("RetrieverService: Failed to load FAISS index for %s: %s", provider, exc)
        else:
            logger.warning(
                "RetrieverService: No FAISS index found at %s. "
                "Context retrieval will return empty results until the index is built.",
                provider_dir,
            )

    # ------------------------------------------------------------------
    # Core retrieval helper (runs in thread executor)
    # ------------------------------------------------------------------

    def _sync_retrieve(self, query: str, provider: str = "openai") -> str:
        """
        Synchronous retrieval logic executed in a thread-pool executor.
        """
        self._ensure_index(provider)

        pipeline = self._pipelines.get(provider)
        if not self._index_loaded.get(provider, False) or pipeline is None:
            logger.warning("RetrieverService: Index not available for %s — returning empty context.", provider)
            return ""

        try:
            chunks = pipeline.retrieve_top_k(
                query,
                k=self.TOP_K,
                score_threshold=self.SCORE_THRESHOLD,
                provider=provider
            )
        except Exception as exc:
            logger.error("RetrieverService: retrieve_top_k failed: %s", exc)
            return ""

        if not chunks:
            logger.info("RetrieverService: No chunks above threshold for query: %r", query[:80])
            return ""

        # Format chunks into a clean numbered context block
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            header = f"[{i}] Source: {chunk.source}  (relevance: {chunk.score:.2f})"
            parts.append(f"{header}\n{chunk.text.strip()}")

        context = "\n\n".join(parts)
        logger.info(
            "RetrieverService: Returning %d chunk(s) for query: %r",
            len(chunks), query[:80],
        )
        return context

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def retrieve_context(self, query: str, provider: str = "openai") -> str:
        """
        Async entry-point called by the orchestrator.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._sync_retrieve, query, provider))


# ---------------------------------------------------------------------------
# Singleton instance for use across routes/services
# ---------------------------------------------------------------------------

retriever_service = RetrieverService()