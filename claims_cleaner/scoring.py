import numpy as np
from typing import List, Callable, Optional
from embedding_utils import EmbeddingModel  # Adjust import per your environment


def compute_embeddings(
        claims: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        cache_path: Optional[str] = None,
        show_progress_bar: bool = False
) -> np.ndarray:
    """
    Encodes a list of claims into embeddings using the specified model and device.

    :param claims: List of textual claims.
    :param model_name: HuggingFace/SBERT model to use for embeddings.
    :param device: 'cpu', 'cuda', 'mps', or None (auto).
    :param cache_path: Optional path to cache embeddings or model.
    :param show_progress_bar: Whether to show a progress bar in encoding.
    :return: A (N, D) numpy array of embeddings, where N is len(claims).
    """
    if not claims:
        return np.array([])

    model = EmbeddingModel(
        model_name=model_name,
        device=device,
        cache_path=cache_path
    )
    embeddings = np.array(model.encode_texts(claims, show_progress_bar=show_progress_bar))
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
