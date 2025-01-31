from typing import List, Dict, Optional

import numpy as np
from embedding_utils import EmbeddingModel

from claim_deduplicator.paths import EmbeddingCachePaths


def compute_embeddings(
        claims: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        show_progress_bar: bool = False
) -> np.ndarray:
    """
    Encodes a list of claims into embeddings with a specified model and device,
    ALWAYS using an on-disk cache:
      1. Load cache if available.
      2. Encode missing texts, store them in the in-memory cache.
      3. Save the updated cache back to disk.

    :param claims: List of textual claims.
    :param model_name: e.g. "sentence-transformers/all-MiniLM-L6-v2".
    :param device: 'cpu', 'cuda', 'mps', or None (auto-detect).
    :param cache_path: Path to .pkl file used for caching. If None, we derive it from EmbeddingCachePaths.
    :param show_progress_bar: Whether to show a progress bar while encoding.
    :return: A numpy array of shape (len(claims), dim).
    """
    if not claims:
        return np.array([])

    # 1) Determine the cache path (if none provided)
    if cache_path is None:
        ecp = EmbeddingCachePaths()
        cache_path = ecp.get_cache_file_for_model(model_name)

    # 2) Initialize EmbeddingModel (which loads cache automatically if it exists)
    model = EmbeddingModel(
        model_name=model_name,
        device=device,
        cache_path=str(cache_path)
    )

    # 3) Encode texts (uses + updates in-memory cache)
    embeddings_list = model.encode_texts(claims, show_progress_bar=show_progress_bar)
    embeddings = np.array(embeddings_list)

    # 4) Save the updated cache to disk *every single time*
    model.save_cache()

    return embeddings


def embed_unique_claims(
    all_claims: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    show_progress_bar: bool = False
) -> Dict[str, np.ndarray]:
    """
    1) deduplicate textual claims to get unique
    2) embed them using EmbeddingModel w/ caching
    3) build {claim: embedding}
    """
    if not all_claims:
        return {}

    unique_claims = list(set(all_claims))
    model_cache_path = EmbeddingCachePaths().get_cache_file_for_model(model_name)

    # Embedding model with on-disk cache
    model = EmbeddingModel(
        model_name=model_name,
        device=device,
        cache_path=str(model_cache_path)
    )

    embeddings_array = model.encode_texts(unique_claims, show_progress_bar=show_progress_bar)
    model.save_cache()

    claim2emb = {}
    for txt, emb in zip(unique_claims, embeddings_array):
        claim2emb[txt] = emb

    return claim2emb
