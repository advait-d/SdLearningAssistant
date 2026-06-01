"""
ingest_docs.py — Build the FAISS index from documents.

Scans a source directory (default: staticdocs/) for supported files,
extracts text, embeds them, and saves the FAISS index to disk.

Usage:
    # Ingest from staticdocs/ (default) using Gemini embeddings
    python -m scripts.ingest_docs --provider gemini

    # Ingest from a custom directory
    python -m scripts.ingest_docs --provider gemini --source /path/to/docs

    # Use OpenAI embeddings (requires paid API key)
    python -m scripts.ingest_docs --provider openai

The index is saved to data/faiss_index/<provider>/ and automatically
loaded by RetrieverService on the next server startup.

Supported file types: .pdf, .md, .txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

# Make sure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env manually (no python-dotenv needed) ──────────────────────────────
def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

_load_env(ROOT / ".env")

# ── Import AFTER env load ─────────────────────────────────────────────────────
from app.services.retriever_service import RAGPipeline, Document  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
# Suppress noisy pypdf warnings about malformed PDFs
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
logger = logging.getLogger(__name__)


# ── File readers ──────────────────────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    try:
        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", path.name, e)
        return ""


def _read_text(path: Path) -> str:
    """Read a plain text or markdown file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path.name, e)
        return ""


def _clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Collapse runs of whitespace/newlines
    - Remove null bytes and non-printable characters
    - Strip leading/trailing whitespace
    """
    # Remove null bytes
    text = text.replace("\x00", " ")
    # Collapse excessive blank lines (3+ newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ── Directory scanner ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_documents_from_dir(source_dir: Path) -> List[Document]:
    """
    Recursively scan source_dir for supported files.
    Returns a list of Document objects ready for ingestion.
    """
    if not source_dir.exists():
        logger.error("Source directory not found: %s", source_dir)
        sys.exit(1)

    files = sorted(
        f for f in source_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.error(
            "No supported files found in %s (looking for: %s)",
            source_dir, ", ".join(SUPPORTED_EXTENSIONS)
        )
        sys.exit(1)

    logger.info("Found %d file(s) in %s", len(files), source_dir)

    documents: List[Document] = []
    for path in files:
        ext = path.suffix.lower()
        logger.info("  Reading: %s", path.name)

        if ext == ".pdf":
            raw = _read_pdf(path)
        else:
            raw = _read_text(path)

        text = _clean_text(raw)

        if len(text.split()) < 50:
            logger.warning("  Skipping %s — too short after extraction (%d words)", path.name, len(text.split()))
            continue

        # Derive a topic tag from the file name for metadata
        stem = path.stem.lower()
        topic = "general"
        if any(w in stem for w in ("redis", "kafka", "queue", "pub", "message")):
            topic = "messaging"
        elif any(w in stem for w in ("kubernetes", "docker", "container", "gitops", "cloud")):
            topic = "infrastructure"
        elif any(w in stem for w in ("linux", "git", "command")):
            topic = "devops"
        elif any(w in stem for w in ("concurrent", "thread", "lock", "parallel", "atomic")):
            topic = "concurrency"
        elif any(w in stem for w in ("machine", "learning", "ml", "reinforcement", "islp", "algorithm")):
            topic = "machine-learning"
        elif any(w in stem for w in ("scalab", "distributed", "system", "network", "soa", "event")):
            topic = "system-design"
        elif any(w in stem for w in ("java", "python", "node", "microprofile", "reactive")):
            topic = "programming"

        documents.append(Document(
            content=text,
            source=path.name,
            metadata={
                "topic": topic,
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
            },
        ))

    logger.info("Loaded %d document(s) with enough content", len(documents))
    return documents


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from documents in a source directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.ingest_docs --provider gemini
  python -m scripts.ingest_docs --provider gemini --source staticdocs
  python -m scripts.ingest_docs --provider openai --source /path/to/my/docs
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="Embedding provider (default: gemini)",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "staticdocs"),
        help="Directory of documents to ingest (default: staticdocs/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "faiss_index"),
        help="Base output directory for the FAISS index",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Target chunk size in words (default: 400)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="Overlap between chunks in words (default: 80)",
    )
    args = parser.parse_args()

    provider   = args.provider
    source_dir = Path(args.source)
    output_dir = Path(args.output_dir) / provider

    # ── Validate API keys ────────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if provider == "gemini" and not gemini_key:
        logger.error("GEMINI_API_KEY not set. Check your .env file.")
        sys.exit(1)
    if provider == "openai" and not openai_key:
        logger.error("OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    # ── Load documents ────────────────────────────────────────────────────────
    documents = load_documents_from_dir(source_dir)

    logger.info(
        "Starting ingestion: provider=%s | docs=%d | chunk_size=%d | overlap=%d",
        provider, len(documents), args.chunk_size, args.overlap,
    )

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = RAGPipeline(api_key=openai_key)

    if provider == "gemini" and gemini_key:
        try:
            from google import genai
            pipeline.gemini_client = genai.Client(api_key=gemini_key)
        except ImportError:
            logger.error("google-genai not installed. Run: pip install google-genai")
            sys.exit(1)

    # ── Ingest ────────────────────────────────────────────────────────────────
    chunks = pipeline.chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    logger.info("Total chunks to embed: %d", len(chunks))

    embeddings = pipeline.create_embeddings(chunks, provider=provider)
    pipeline.store_in_faiss(chunks, embeddings)
    pipeline.save_index(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Index saved to: %s", output_dir)
    logger.info("Vectors:        %d", pipeline.index.ntotal)
    logger.info("Dimension:      %d", embeddings.shape[1])
    logger.info("=" * 60)
    logger.info("Done! Restart the server to load the new index.")


if __name__ == "__main__":
    main()
