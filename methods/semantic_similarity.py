import hashlib
import math
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def build_semantic_hybrid_similarity(item_metadata,
                                     num_items: int,
                                     candidate_pools: Dict[int, Sequence],
                                     top_k: int = 5,
                                     cache_dir: str = "cache",
                                     hashing_dim: int = 4096) -> Dict[int, List[Tuple[int, float]]]:
    """Rerank co-occurrence pools with lightweight lexical semantic vectors.

    This avoids sentence-transformer inference. Item texts are represented by
    hashed TF-IDF vectors, and cosine is computed only inside each source
    item's co-occurrence candidate pool.
    """
    item_ids, texts = _extract_item_texts(item_metadata, num_items)
    if len(item_ids) <= 1:
        return {}

    hashing_dim = max(128, int(hashing_dim))
    top_k = max(1, int(top_k))
    text_hash = _hash_texts(item_ids, texts, f"hash_tfidf:{hashing_dim}")
    pool_hash = _hash_candidate_pools(candidate_pools)
    cache_key = _cache_key(f"hash_tfidf_{hashing_dim}_{pool_hash}", num_items, top_k, text_hash)
    os.makedirs(cache_dir, exist_ok=True)
    candidates_path = os.path.join(cache_dir, f"semantic_hybrid_candidates_{cache_key}.pkl")

    if os.path.exists(candidates_path):
        with open(candidates_path, "rb") as handle:
            return pickle.load(handle)

    vectors = _build_hashed_tfidf_vectors(item_ids, texts, hashing_dim)
    candidates: Dict[int, List[Tuple[int, float]]] = {}
    fallback = _rerank_candidate_pool(0, candidate_pools.get(0, []), vectors, top_k, num_items)
    if fallback:
        candidates[0] = fallback

    for source_item, pool in candidate_pools.items():
        try:
            source_item = int(source_item)
        except (TypeError, ValueError):
            continue
        if source_item <= 0 or source_item > num_items:
            continue
        ranked = _rerank_candidate_pool(source_item, pool, vectors, top_k, num_items)
        if ranked:
            candidates[source_item] = ranked

    with open(candidates_path, "wb") as handle:
        pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return candidates


def build_semantic_embedding_similarity(item_metadata,
                                        num_items: int,
                                        top_k: int = 20,
                                        model_name: str = "BAAI/bge-m3",
                                        cache_dir: str = "cache",
                                        encode_batch_size: int = 128,
                                        retrieval_batch_size: int = 512,
                                        device: Optional[str] = None,
                                        search_device: Optional[str] = None,
                                        text_prefix: Optional[str] = None) -> Dict[int, List[Tuple[int, float]]]:
    """Build item replacement candidates from semantic text embeddings.

    The returned dictionary maps source item ids to [(candidate_id, cosine)].
    Item texts come from the processed items table and are encoded once, then
    cached under cache_dir.
    """
    item_ids, texts = _extract_item_texts(item_metadata, num_items)
    if len(item_ids) <= 1:
        return {}

    resolved_prefix = _resolve_text_prefix(model_name, text_prefix)
    text_hash = _hash_texts(item_ids, texts, resolved_prefix)
    cache_key = _cache_key(model_name, num_items, top_k, text_hash)
    os.makedirs(cache_dir, exist_ok=True)
    candidates_path = os.path.join(cache_dir, f"semantic_candidates_{cache_key}.pkl")
    embeddings_path = os.path.join(cache_dir, f"semantic_embeddings_{cache_key}.pt")

    if os.path.exists(candidates_path):
        with open(candidates_path, "rb") as handle:
            return pickle.load(handle)

    if os.path.exists(embeddings_path):
        payload = torch.load(embeddings_path, map_location="cpu")
        embeddings = payload["embeddings"].float()
        cached_item_ids = [int(item) for item in payload["item_ids"]]
        if cached_item_ids != item_ids:
            embeddings = _encode_texts(texts, model_name, encode_batch_size, device, resolved_prefix)
            torch.save({"item_ids": item_ids, "embeddings": embeddings}, embeddings_path)
    else:
        embeddings = _encode_texts(texts, model_name, encode_batch_size, device, resolved_prefix)
        torch.save({"item_ids": item_ids, "embeddings": embeddings}, embeddings_path)

    candidates = _semantic_topk(
        item_ids=item_ids,
        embeddings=embeddings,
        top_k=top_k,
        batch_size=retrieval_batch_size,
        search_device=search_device,
    )
    with open(candidates_path, "wb") as handle:
        pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return candidates


def _build_hashed_tfidf_vectors(item_ids: Sequence[int],
                                texts: Sequence[str],
                                hashing_dim: int) -> Dict[int, Dict[int, float]]:
    term_counts_by_item: Dict[int, Dict[int, float]] = {}
    doc_freq = defaultdict(int)

    for item_id, text in zip(item_ids, texts):
        counts = defaultdict(float)
        for token in _tokenize_text(text):
            feature_id = _stable_token_hash(token) % hashing_dim
            counts[feature_id] += 1.0
        if not counts:
            continue
        item_id = int(item_id)
        term_counts_by_item[item_id] = dict(counts)
        for feature_id in counts:
            doc_freq[feature_id] += 1

    n_docs = max(1, len(term_counts_by_item))
    vectors: Dict[int, Dict[int, float]] = {}
    for item_id, term_counts in term_counts_by_item.items():
        weighted = {}
        norm_sq = 0.0
        for feature_id, count in term_counts.items():
            tf = 1.0 + math.log(float(count))
            idf = math.log((1.0 + n_docs) / (1.0 + doc_freq[feature_id])) + 1.0
            value = tf * idf
            weighted[feature_id] = value
            norm_sq += value * value
        if norm_sq <= 0.0:
            continue
        norm = math.sqrt(norm_sq)
        vectors[item_id] = {
            feature_id: value / norm
            for feature_id, value in weighted.items()
        }
    return vectors


def _rerank_candidate_pool(source_item: int,
                           pool: Sequence,
                           vectors: Dict[int, Dict[int, float]],
                           top_k: int,
                           num_items: int) -> List[Tuple[int, float]]:
    source_vector = vectors.get(int(source_item), {})
    scored = []
    seen = set()
    for rank, raw_candidate in enumerate(pool):
        candidate_id = _candidate_id(raw_candidate)
        if candidate_id is None:
            continue
        if candidate_id <= 0 or candidate_id > num_items or candidate_id == source_item:
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        score = _sparse_dot(source_vector, vectors.get(candidate_id, {}))
        scored.append((candidate_id, max(0.0, min(1.0, score)), rank))

    scored.sort(key=lambda kv: (-kv[1], kv[2]))
    return [
        (candidate_id, score)
        for candidate_id, score, _ in scored[:top_k]
    ]


def _candidate_id(raw_candidate) -> Optional[int]:
    if isinstance(raw_candidate, dict):
        raw_candidate = raw_candidate.get("candidate", raw_candidate.get("item", raw_candidate.get("id")))
    elif isinstance(raw_candidate, (tuple, list)) and raw_candidate:
        raw_candidate = raw_candidate[0]
    try:
        return int(raw_candidate)
    except (TypeError, ValueError):
        return None


def _sparse_dot(left: Dict[int, float], right: Dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(feature_id, 0.0) for feature_id, value in left.items()))


def _tokenize_text(text: str) -> List[str]:
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)


def _stable_token_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _extract_item_texts(item_metadata, num_items: int) -> Tuple[List[int], List[str]]:
    if item_metadata is None:
        return [], []

    item_ids: List[int] = []
    texts: List[str] = []
    if hasattr(item_metadata, "iterrows"):
        for _, row in item_metadata.iterrows():
            item_id = _row_get(row, "movie_id", _row_get(row, "item_id", None))
            text = _row_get(row, "item_text", None)
            if text is None or not str(text).strip() or str(text).lower() == "nan":
                text = _build_fallback_text(row)
            try:
                item_id = int(item_id)
            except (TypeError, ValueError):
                continue
            text = str(text).strip()
            if 0 < item_id <= num_items and text:
                item_ids.append(item_id)
                texts.append(text)
    return item_ids, texts


def _row_get(row, key: str, default=None):
    try:
        return row.get(key, default)
    except AttributeError:
        return default


def _build_fallback_text(row) -> str:
    parts = []
    for key in ("title", "genres", "artist_name", "track_name", "cate_id", "brand", "price"):
        value = _row_get(row, key, None)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            parts.append(f"{key}: {value}")
    return ". ".join(parts)


def _resolve_text_prefix(model_name: str, text_prefix: Optional[str]) -> str:
    if text_prefix is not None:
        return str(text_prefix)
    return "passage: " if "e5" in str(model_name).lower() else ""


def _encode_texts(texts: Sequence[str],
                  model_name: str,
                  batch_size: int,
                  device: Optional[str],
                  text_prefix: str) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic embedding similarity requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=model_device)
    encoded_texts = [f"{text_prefix}{text}" for text in texts]
    try:
        embeddings = model.encode(
            encoded_texts,
            batch_size=int(batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except TypeError:
        embeddings = model.encode(
            encoded_texts,
            batch_size=int(batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)


def _semantic_topk(item_ids: Sequence[int],
                   embeddings: torch.Tensor,
                   top_k: int,
                   batch_size: int,
                   search_device: Optional[str]) -> Dict[int, List[Tuple[int, float]]]:
    item_ids = [int(item) for item in item_ids]
    embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    n_items = embeddings.size(0)
    k = min(int(top_k), max(0, n_items - 1))
    if k <= 0:
        return {0: []}

    device = torch.device(search_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_embeddings = embeddings.to(device)
    result: Dict[int, List[Tuple[int, float]]] = {
        0: [(item_id, 0.0) for item_id in item_ids[:k]]
    }

    batch_size = max(1, int(batch_size))
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        block = all_embeddings[start:end]
        scores = torch.matmul(block, all_embeddings.t())
        diag_rows = torch.arange(end - start, device=device)
        diag_cols = torch.arange(start, end, device=device)
        scores[diag_rows, diag_cols] = float("-inf")

        values, indices = torch.topk(scores, k=k, dim=1)
        values = values.detach().cpu().tolist()
        indices = indices.detach().cpu().tolist()

        for offset, (row_values, row_indices) in enumerate(zip(values, indices)):
            source_id = item_ids[start + offset]
            candidates = []
            for score, neighbor_idx in zip(row_values, row_indices):
                if not np.isfinite(score):
                    continue
                candidate_id = item_ids[int(neighbor_idx)]
                if candidate_id == source_id:
                    continue
                candidates.append((candidate_id, max(0.0, min(1.0, float(score)))))
            if candidates:
                result[source_id] = candidates

    return result


def _hash_texts(item_ids: Sequence[int], texts: Sequence[str], text_prefix: str) -> str:
    digest = hashlib.sha1()
    digest.update(text_prefix.encode("utf-8"))
    for item_id, text in zip(item_ids, texts):
        digest.update(f"{int(item_id)}\t{text}\n".encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


def _hash_candidate_pools(candidate_pools: Dict[int, Sequence]) -> str:
    digest = hashlib.sha1()
    for source_item in sorted(candidate_pools.keys(), key=lambda item: int(item)):
        digest.update(f"{int(source_item)}:".encode("utf-8"))
        for raw_candidate in candidate_pools[source_item]:
            candidate_id = _candidate_id(raw_candidate)
            if candidate_id is not None:
                digest.update(f"{candidate_id},".encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _cache_key(model_name: str, num_items: int, top_k: int, text_hash: str) -> str:
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("_")
    return f"{model_key}_n{int(num_items)}_k{int(top_k)}_{text_hash}"
