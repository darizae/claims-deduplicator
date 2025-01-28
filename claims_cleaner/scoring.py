from collections import deque

import numpy as np
from typing import List, Callable, Optional, Dict
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


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two embedding vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def build_similarity_matrix(
        embeddings: np.ndarray,
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] = compute_cosine_similarity
) -> np.ndarray:
    """
    Given an (N, D) embeddings array, compute an (N, N) similarity matrix
    using the provided similarity function.

    :param embeddings: 2D array of shape (N, D).
    :param similarity_fn: A function that takes (vec1, vec2) and returns a float [0..1].
    :return: A 2D numpy array (N, N), sim_matrix[i][j] = similarity_fn(embeddings[i], embeddings[j]).
    """
    n = len(embeddings)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            sim = similarity_fn(embeddings[i], embeddings[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix


def build_similarity_matrix_vectorized(embeddings: np.ndarray) -> np.ndarray:
    """
    Given an (N, D) array of embeddings,
    compute the NxN cosine similarity matrix in a single step:
        sim_matrix = (E / ||E||) dot (E / ||E||)^T
    """
    # 1) Normalize embeddings along axis=1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-9  # avoid division by zero
    normed_embs = embeddings / norms

    # 2) Use matrix multiplication for pairwise dot products
    sim_matrix = normed_embs @ normed_embs.T  # shape (N, N)
    return sim_matrix


def build_adjacency_from_matrix(sim_matrix: np.ndarray, threshold: float) -> dict:
    """
    Build an adjacency list (i -> list of connected j's) from a similarity matrix
    given a threshold. If sim >= threshold, there's an edge between i and j.

    :param sim_matrix: 2D array of shape (N, N).
    :param threshold: Cosine similarity threshold to consider "duplicates."
    :return: A dict {i: [j, ...], ...} denoting edges.
    """
    n = sim_matrix.shape[0]
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def build_adjacency_from_sim_matrix(sim_matrix: np.ndarray, threshold: float) -> Dict[int, List[int]]:
    """
    Construct adjacency from NxN sim_matrix, where sim_matrix[i,j] >= threshold => edge i<->j.
    """
    n = sim_matrix.shape[0]
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        # We only need to loop j > i to avoid duplication
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency
