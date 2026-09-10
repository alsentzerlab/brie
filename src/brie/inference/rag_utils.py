"""
Shared utilities for RAG-based inference scripts.

Provides:
  - Note chunking (character-based, sized for Octen/Octen-Embedding-8B and LateOn)
  - Per-patient embedding cache (pickle, keyed by patient_id)
  - BM25, dense, hybrid+reranker, and late-interaction (ColBERT/LateOn) retrieval
  - Chunk formatting and serialisation for CSV output
"""

import json
import os
import pickle
import re
from typing import Any

import numpy as np

# dependencies
# sentence transformers
# pylate
# rank bm25


# ── Constants ─────────────────────────────────────────────────────────────────
# Octen-Embedding-8B supports up to 40 960 tokens per sequence.
# 4 000-char chunks (~1 000 tokens) give dense, coherent retrieval units
# while leaving ample headroom for the 40K context.
CHUNK_SIZE     = 4000   # characters per chunk
CHUNK_OVERLAP  = 400    # character overlap between consecutive chunks
TOP_K          = 10     # default number of chunks to retrieve
CANDIDATE_K    = 100    # hybrid pre-filter pool before reranking
RRF_K          = 60     # RRF constant

EMBEDDING_MODEL = os.environ.get("BRIE_EMBEDDING_MODEL", "")

RERANKER_MODEL = os.environ.get(
    "BRIE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Octen-Embedding-8B: no query prefix; passages prepend "- " to avoid
# upstream tokenizer issues inherited from Qwen3-Embedding.
_OCTEN_DOC_PREFIX = "- "

LATE_MODEL = os.environ.get("BRIE_LATE_INTERACTION_MODEL", "")


# ── LateOn (ColBERT late-interaction) ─────────────────────────────────────────
LATE_CHUNK_SIZE    = 800   # characters per chunk
LATE_CHUNK_OVERLAP = 80    # character overlap between chunks

_colbert_model: Any = None

# ── Lazy model singletons ─────────────────────────────────────────────────────
_st_model:      Any = None
_cross_encoder: Any = None


def _pick_gpu(min_free_gb: float = 20.0) -> str:
    """Return the cuda device with the most free memory (if >= min_free_gb), else 'cpu'."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    best_idx, best_free = 0, 0.0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free_gb = free / 1024 ** 3
        if free_gb > best_free:
            best_free, best_idx = free_gb, i
    if best_free < min_free_gb:
        return "cpu"
    return f"cuda:{best_idx}"


def _get_st_model() -> Any:
    global _st_model
    if _st_model is None:
        if not EMBEDDING_MODEL:
            raise RuntimeError("Set BRIE_EMBEDDING_MODEL before embedding retrieval")
        from sentence_transformers import SentenceTransformer
        device = os.environ.get("EMBEDDING_DEVICE") or _pick_gpu()
        # Restrict the process to a single GPU so accelerate doesn't spread the
        # model across all visible devices via device_map="auto".
        if device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
            device = "cuda:0"
        _st_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        _st_model.max_seq_length = 2048 # truncate since chunks are 4000 characters
    return _st_model


def _get_cross_encoder() -> Any:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


# ── Chunking ──────────────────────────────────────────────────────────────────
def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character splitter: tries paragraph → line → sentence → word boundaries."""
    for sep in ["\n\n", "\n", ". ", " "]:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            piece = part + sep
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                # carry overlap from the end of the previous chunk
                current = current[max(0, len(current) - overlap):] + piece
        if current.strip():
            chunks.append(current.strip())
        if chunks:
            return chunks

    # Fallback: hard character split
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), chunk_size - overlap)
        if text[i : i + chunk_size].strip()
    ]


def chunk_patient_notes(records: list[dict], patient_id: str) -> list[dict]:
    """
    Chunk all notes for a patient into retrieval units.

    Each chunk dict contains:
      chunk_id, patient_id, note_idx, note_title, note_date, text
    """
    chunks: list[dict] = []
    for note_idx, note in enumerate(records):
        text_chunks = _split_text(note.get("text", ""), CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id":   f"{patient_id}_{note_idx}_{chunk_idx}",
                "patient_id": patient_id,
                "note_idx":   note_idx,
                "note_title": note.get("note_title", ""),
                "note_date":  str(note.get("note_date", "")),
                "text":       chunk_text,
            })
    return chunks


# ── Embedding cache ────────────────────────────────────────────────────────────
def _cache_path(embeddings_dir: str, patient_id: str) -> str:
    return os.path.join(embeddings_dir, f"{patient_id}_embeddings.pkl")


def _save_patient_embeddings(
    path: str,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings.tolist()}, f)


def load_cached_embeddings(
    patient_id: str,
    embeddings_dir: str,
) -> tuple[list[dict], np.ndarray] | None:
    """Return cached (chunks, embeddings) for a patient, or None if not cached."""
    path = _cache_path(embeddings_dir, patient_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    return cached["chunks"], np.array(cached["embeddings"], dtype=np.float32)


def build_embeddings_batch(
    uncached: dict[str, list[dict]],
    embeddings_dir: str,
) -> dict[str, tuple[list[dict], np.ndarray]]:
    """
    Encode and cache embeddings for multiple patients in a single model.encode() call.

    All chunk texts across all patients are concatenated into one list so the
    encoder processes them in proper batches (e.g. 5/12) rather than one
    patient at a time (1/1).  Per-patient pickle files are saved and the
    resulting {patient_id: (chunks, embeddings)} dict is returned.
    """
    if not uncached:
        return {}

    os.makedirs(embeddings_dir, exist_ok=True)

    # Build per-patient chunks and record slice boundaries in the flat list
    all_chunks:  list[dict] = []
    all_texts:   list[str]  = []
    boundaries:  dict[str, tuple[int, int]] = {}  # patient_id -> (start, end)
    patient_chunks: dict[str, list[dict]] = {}

    for patient_id, records in uncached.items():
        chunks = chunk_patient_notes(records, patient_id)
        start  = len(all_texts)
        all_texts.extend(c["text"] for c in chunks)
        boundaries[patient_id]    = (start, len(all_texts))
        patient_chunks[patient_id] = chunks
        all_chunks.extend(chunks)

    model = _get_st_model()
    prefixed_texts = [_OCTEN_DOC_PREFIX + t for t in all_texts]
    all_embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    ).astype(np.float32)

    result: dict[str, tuple[list[dict], np.ndarray]] = {}
    for patient_id, (start, end) in boundaries.items():
        chunks     = patient_chunks[patient_id]
        embeddings = all_embeddings[start:end]
        _save_patient_embeddings(
            _cache_path(embeddings_dir, patient_id), chunks, embeddings
        )
        result[patient_id] = (chunks, embeddings)

    return result


def load_or_build_embeddings(
    records: list[dict],
    patient_id: str,
    embeddings_dir: str,
) -> tuple[list[dict], np.ndarray]:
    """Single-patient fallback. Prefer build_embeddings_batch() for multiple patients."""
    cached = load_cached_embeddings(patient_id, embeddings_dir)
    if cached is not None:
        return cached

    result = build_embeddings_batch({patient_id: records}, embeddings_dir)
    return result[patient_id]


# ── Retrieval ─────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_bm25(
    query: str,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """BM25Okapi retrieval over pre-chunked notes. Returns top_k chunks with scores."""
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


def encode_queries(queries: list[str], batch_size: int | None = None) -> np.ndarray:
    """
    Encode a list of queries in one batched model.encode() call.

    Returns an (N, D) float32 array of L2-normalised query embeddings.
    Call this once before the inference loop and pass slices to the
    retrieve_*_precomputed() functions to avoid per-task model.encode() calls.
    """
    model = _get_st_model()
    if batch_size is None:
        batch_size = int(os.environ.get("QUERY_EMBEDDING_BATCH_SIZE", "32"))
    return model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype(np.float32)


def retrieve_embedding(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """Dense cosine-similarity retrieval (embeddings must be L2-normalised)."""
    model     = _get_st_model()
    query_emb = model.encode(
        query,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores  = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


def retrieve_embedding_precomputed(
    query_emb: np.ndarray,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """Dense retrieval using a pre-computed, L2-normalised query embedding."""
    scores  = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


def retrieve_hybrid(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:
    """
    Hybrid retrieval: RRF fusion of BM25 + dense rankings, then cross-encoder reranking.

    Steps:
      1. BM25 ranks all chunks.
      2. Dense cosine-similarity ranks all chunks.
      3. Reciprocal Rank Fusion combines both ranked lists.
      4. Top candidate_k from RRF are reranked by a cross-encoder.
      5. Top top_k returned.
    """
    from rank_bm25 import BM25Okapi

    n = len(chunks)

    # BM25 ranks
    corpus     = [_tokenize(c["text"]) for c in chunks]
    bm25       = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_order  = np.argsort(bm25_scores)[::-1]
    bm25_rank   = {int(idx): rank for rank, idx in enumerate(bm25_order)}

    # Dense ranks
    model     = _get_st_model()
    query_emb = model.encode(
        query,
        normalize_embeddings=True,
    ).astype(np.float32)
    dense_scores = embeddings @ query_emb
    dense_order  = np.argsort(dense_scores)[::-1]
    dense_rank   = {int(idx): rank for rank, idx in enumerate(dense_order)}

    # RRF fusion
    rrf = {
        i: (1.0 / (RRF_K + bm25_rank.get(i, n)) +
            1.0 / (RRF_K + dense_rank.get(i, n)))
        for i in range(n)
    }
    candidates = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:min(candidate_k, n)]

    # Cross-encoder reranking
    reranker = _get_cross_encoder()
    pairs    = [(query, chunks[i]["text"]) for i in candidates]
    ce_scores = reranker.predict(pairs)
    top_local = np.argsort(ce_scores)[::-1][:top_k]
    return [
        {**chunks[candidates[j]], "score": float(ce_scores[j])}
        for j in top_local
    ]


def retrieve_hybrid_precomputed(
    query: str,
    query_emb: np.ndarray,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:
    """Hybrid retrieval using a pre-computed, L2-normalised query embedding."""
    candidates, pairs = get_rrf_candidates(query, query_emb, chunks, embeddings, candidate_k)
    reranker  = _get_cross_encoder()
    ce_scores = reranker.predict(pairs)
    return rerank_from_scores(ce_scores, candidates, chunks, top_k)


def get_rrf_candidates(
    query: str,
    query_emb: np.ndarray,
    chunks: list[dict],
    embeddings: np.ndarray,
    candidate_k: int = CANDIDATE_K,
) -> tuple[list[int], list[tuple[str, str]]]:
    """
    Run BM25 + dense + RRF fusion and return the top candidate_k chunk indices
    and their (query, chunk_text) pairs ready for cross-encoder scoring.

    Call this for every task upfront, then batch all pairs through
    encode_reranker_pairs() before the async inference loop.
    """
    from rank_bm25 import BM25Okapi

    n = len(chunks)

    corpus      = [_tokenize(c["text"]) for c in chunks]
    bm25        = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_order  = np.argsort(bm25_scores)[::-1]
    bm25_rank   = {int(idx): rank for rank, idx in enumerate(bm25_order)}

    dense_scores = embeddings @ query_emb
    dense_order  = np.argsort(dense_scores)[::-1]
    dense_rank   = {int(idx): rank for rank, idx in enumerate(dense_order)}

    rrf = {
        i: (1.0 / (RRF_K + bm25_rank.get(i, n)) +
            1.0 / (RRF_K + dense_rank.get(i, n)))
        for i in range(n)
    }
    candidates = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:min(candidate_k, n)]
    pairs = [(query, chunks[i]["text"]) for i in candidates]
    return candidates, pairs


def encode_reranker_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    Score all (query, chunk_text) pairs in one reranker.predict() call.

    Collect pairs from every task with get_rrf_candidates(), concatenate them,
    call this once, then slice results back per task with rerank_from_scores().
    """
    reranker = _get_cross_encoder()
    return reranker.predict(pairs, show_progress_bar=True)


def rerank_from_scores(
    ce_scores: np.ndarray,
    candidates: list[int],
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """Return top_k chunks using pre-computed cross-encoder scores."""
    top_local = np.argsort(ce_scores)[::-1][:top_k]
    return [
        {**chunks[candidates[j]], "score": float(ce_scores[j])}
        for j in top_local
    ]


def _get_colbert_model() -> Any:
    global _colbert_model
    if _colbert_model is None:
        if not LATE_MODEL:
            raise RuntimeError("Set BRIE_LATE_INTERACTION_MODEL before late-interaction retrieval")
        from pylate import models
        _colbert_model = models.ColBERT(model_name_or_path=LATE_MODEL, trust_remote_code=True, device=_pick_gpu())
    return _colbert_model


def chunk_patient_notes_late(records: list[dict], patient_id: str) -> list[dict]:
    """Chunk notes into LateOn-sized units (LATE_CHUNK_SIZE chars)."""
    chunks: list[dict] = []
    for note_idx, note in enumerate(records):
        text_chunks = _split_text(note.get("text", ""), LATE_CHUNK_SIZE, LATE_CHUNK_OVERLAP)
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id":   f"{patient_id}_{note_idx}_{chunk_idx}",
                "patient_id": patient_id,
                "note_idx":   note_idx,
                "note_title": note.get("note_title", ""),
                "note_date":  str(note.get("note_date", "")),
                "text":       chunk_text,
            })
    return chunks


def _late_embs_path(embeddings_dir: str, patient_id: str) -> str:
    return os.path.join(embeddings_dir, f"{patient_id}_late_embs.pkl")


def load_cached_late_embeddings(
    patient_id: str,
    embeddings_dir: str,
) -> tuple[list[dict], list[np.ndarray]] | None:
    """Return cached (chunks, doc_embeddings) for a patient, or None if not cached."""
    path = _late_embs_path(embeddings_dir, patient_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    return cached["chunks"], cached["embeddings"]


def build_late_embeddings_batch(
    uncached: dict[str, list[dict]],
    embeddings_dir: str,
) -> dict[str, tuple[list[dict], list[np.ndarray]]]:
    """
    Encode documents for per-patient ColBERT retrieval and cache as numpy arrays.
    """
    if not uncached:
        return {}

    os.makedirs(embeddings_dir, exist_ok=True)

    all_texts:      list[str]  = []
    boundaries:     dict[str, tuple[int, int]] = {}
    patient_chunks: dict[str, list[dict]]      = {}

    for patient_id, records in uncached.items():
        chunks = chunk_patient_notes_late(records, patient_id)
        start  = len(all_texts)
        all_texts.extend(c["text"] for c in chunks)
        boundaries[patient_id]     = (start, len(all_texts))
        patient_chunks[patient_id] = chunks

    model = _get_colbert_model()
    all_embeddings = model.encode(
        all_texts,
        is_query=False,
        batch_size=32,
        show_progress_bar=True,
    )

    result: dict[str, tuple[list[dict], list[np.ndarray]]] = {}
    for patient_id, (start, end) in boundaries.items():
        chunks = patient_chunks[patient_id]
        embs   = all_embeddings[start:end]

        path = _late_embs_path(embeddings_dir, patient_id)
        with open(path, "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": embs}, f)

        result[patient_id] = (chunks, embs)

    return result


def encode_queries_late(queries: list[str]) -> list[np.ndarray]:
    """Encode queries with LateOn. Returns list of [num_tokens, 128] float32 arrays."""
    model = _get_colbert_model()
    return model.encode(
        queries,
        is_query=True,
        batch_size=256,
        show_progress_bar=True,
    )


def retrieve_late_precomputed(
    query_emb: np.ndarray,
    chunks: list[dict],
    doc_embeddings: list[np.ndarray],
    top_k: int = TOP_K,
) -> list[dict]:
    """ColBERT MaxSim retrieval over pre-computed document token embeddings."""
    scores = np.array([
        float(np.dot(query_emb, doc_emb.T).max(axis=1).sum())
        for doc_emb in doc_embeddings
    ], dtype=np.float32)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_indices]


# ── Output helpers ────────────────────────────────────────────────────────────
def format_retrieved_notes(retrieved: list[dict]) -> str:
    """Format retrieved chunks for inclusion in a prompt."""
    parts = []
    for i, chunk in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] Note Title: {chunk['note_title']}\n"
            f"    Note Date:  {chunk['note_date']}\n"
            f"    Text: {chunk['text']}"
        )
    return "\n\n".join(parts)


def serialise_retrieved(retrieved: list[dict]) -> str:
    """Serialise retrieved chunk metadata to a JSON string for CSV storage."""
    return json.dumps([
        {
            "chunk_id":   c["chunk_id"],
            "note_title": c["note_title"],
            "note_date":  c["note_date"],
            "text":       c["text"],
            "score":      c.get("score"),
        }
        for c in retrieved
    ])
