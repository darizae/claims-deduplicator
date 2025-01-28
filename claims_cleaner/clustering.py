import numpy as np
from collections import deque
from typing import List, Dict, Optional

from .scoring import (
    compute_embeddings,
    build_similarity_matrix,
    build_adjacency_from_matrix, build_similarity_matrix_vectorized, build_adjacency_from_sim_matrix,
)


def bfs_clusters(adjacency: Dict[int, List[int]]) -> List[List[int]]:
    """
    Perform BFS to find connected components in an adjacency list.
    Returns a list of clusters (each cluster is a list of indices).
    """
    visited = set()
    clusters = []

    for i in adjacency:
        if i not in visited:
            queue = deque([i])
            cluster = []
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            clusters.append(cluster)

    return clusters


def cluster_claims(
        claims: List[str],
        threshold: float = 0.85,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        show_progress_bar: bool = False
) -> List[List[int]]:
    """
    Clusters claims by:
      1) Embedding them (ALWAYS using an on-disk cache).
      2) Building a similarity matrix.
      3) Building an adjacency list (threshold-based).
      4) BFS to find connected components.

    Returns a list of clusters, each cluster is a list of claim indices.

    :param claims: List of textual claims
    :param threshold: Cosine similarity threshold for duplicates
    :param model_name: e.g. "sentence-transformers/all-MiniLM-L6-v2"
    :param device: 'cpu', 'cuda', 'mps', or None
    :param cache_path: If None, we auto-resolve from EmbeddingCachePaths
    :param show_progress_bar: Show progress bar for embeddings
    :return: A list of clusters
    """
    if not claims:
        return []

    # 1) ALWAYS calls compute_embeddings => always uses + saves to cache
    embeddings = compute_embeddings(
        claims,
        model_name=model_name,
        device=device,
        cache_path=cache_path,
        show_progress_bar=show_progress_bar
    )

    if embeddings.size == 0:
        return [[i] for i in range(len(claims))]

    # 2) Similarity matrix -> adjacency -> BFS
    sim_matrix = build_similarity_matrix(embeddings)
    adjacency = build_adjacency_from_matrix(sim_matrix, threshold)
    clusters = bfs_clusters(adjacency)
    return clusters


def cluster_claims_in_record(
        record_claims: List[str],
        claim2emb: Dict[str, np.ndarray],
        threshold: float
) -> List[List[int]]:
    """
    1) Retrieve embeddings for each claim in the record from claim2emb (no re-embedding).
    2) Build NxN similarity matrix via vectorized approach.
    3) Build adjacency & BFS to find clusters.
    :return: A list of clusters, each cluster is a list of indices (0..N-1).
    """
    if not record_claims:
        return []

    # 1) Gather embeddings for the record's claims in order
    # We'll keep them in a list to preserve indexing
    embs = np.array([claim2emb[txt] for txt in record_claims])

    # 2) Vectorized similarity
    sim_matrix = build_similarity_matrix_vectorized(embs)

    # 3) Adjacency + BFS
    adjacency = build_adjacency_from_sim_matrix(sim_matrix, threshold)
    clusters = bfs_clusters(adjacency)
    return clusters
