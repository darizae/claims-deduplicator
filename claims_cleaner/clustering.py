import numpy as np
from collections import deque
from typing import List, Dict

from .scoring import (
    build_similarity_matrix_vectorized,
    build_adjacency_from_sim_matrix,
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
    embs = np.array([claim2emb[txt] for txt in record_claims])

    # 2) Vectorized similarity
    sim_matrix = build_similarity_matrix_vectorized(embs)

    # 3) Adjacency + BFS
    adjacency = build_adjacency_from_sim_matrix(sim_matrix, threshold)
    clusters = bfs_clusters(adjacency)

    return clusters
