import numpy as np
from collections import deque
from typing import List, Dict, Optional

from .scoring import (
    compute_embeddings,
    build_similarity_matrix,
    build_adjacency_from_matrix,
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
