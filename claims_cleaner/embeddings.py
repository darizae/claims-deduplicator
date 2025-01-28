from typing import List, Dict, Optional

import numpy as np
from embedding_utils import EmbeddingModel

from claims_cleaner.paths import EmbeddingCachePaths


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
        device: str = None
) -> Dict[str, np.ndarray]:
    """
    Given a big list of claims (with duplicates),
    1) Convert to a set of unique claims,
    2) Embed them in one shot,
    3) Return a dict {claim_text: embedding_vector}.
    """

    unique_claims = list(set(all_claims))  # deduplicate text
    print(f"Total claims: {len(all_claims)}; unique: {len(unique_claims)}")

    # EMBED everything in a single batch
    all_embeddings = compute_embeddings(
        unique_claims,
        model_name=model_name,
        device=device,
        show_progress_bar=True
    )

    # Build the dictionary
    claim2emb = {}
    for text, emb in zip(unique_claims, all_embeddings):
        claim2emb[text] = emb

    return claim2emb
