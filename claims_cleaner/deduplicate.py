import numpy as np
from embedding_utils import EmbeddingModel
from collections import deque
from typing import List, Callable, Optional


def build_adjacency(embeddings: np.ndarray, threshold: float) -> dict:
    """
    Given embeddings and a threshold, build an adjacency list
    (i.e., which indices are near-duplicates with which other indices).
    """
    n = len(embeddings)
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            # Compute cosine similarity
            sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def find_clusters(adjacency: dict) -> List[List[int]]:
    """
    Using BFS, find connected components (clusters) from the adjacency list.
    """
    visited = set()
    clusters = []
    for i in adjacency:
        if i not in visited:
            queue = deque([i])
            cluster = []
            while queue:
                cur = queue.popleft()
                if cur not in visited:
                    visited.add(cur)
                    cluster.append(cur)
                    for neighbor in adjacency[cur]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            clusters.append(cluster)
    return clusters


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def deduplicate_claims(
        claims: List[str],
        threshold: float = 0.85,
        representative_selector: Callable[[List[str]], str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        cache_path: Optional[str] = None
) -> List[str]:
    """
    Clusters near-duplicate claims (cosine similarity >= threshold),
    then selects one representative from each cluster.

    :param claims: List of textual claims.
    :param threshold: Cosine similarity threshold for 'duplicate'.
    :param representative_selector: Function that picks 1 claim from a cluster.
                                    Default picks the "longest" claim.
    :param model_name: Embeddings model to use.
    :param device: 'cpu', 'cuda', 'mps', or None (auto).
    :param cache_path: Optional path for caching embeddings.
    :return: A list of deduplicated/representative claims.
    """
    if not claims:
        return []

    # 1) Initialize the embedding model
    model = EmbeddingModel(
        model_name=model_name,
        device=device,
        cache_path=cache_path
    )

    # 2) Encode claims into embeddings
    embeddings = model.encode_texts(claims, show_progress_bar=False)

    # 3) Build adjacency map
    adjacency = build_adjacency(embeddings, threshold)

    # 4) Cluster using BFS
    clusters = find_clusters(adjacency)

    # 5) Pick a representative for each cluster
    if representative_selector is None:
        # Default: pick "longest" claim
        def representative_selector(cluster_texts: List[str]) -> str:
            return max(cluster_texts, key=len)

    deduped = []
    for cluster_indices in clusters:
        cluster_texts = [claims[idx] for idx in cluster_indices]
        rep = representative_selector(cluster_texts)
        deduped.append(rep)

    return deduped
