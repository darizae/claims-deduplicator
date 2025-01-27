from typing import List, Callable, Optional

from .scoring import (
    compute_embeddings,
    build_similarity_matrix,
    build_adjacency_from_matrix,
)
from .clustering import bfs_clusters
from .strategies import select_longest


def deduplicate_claims(
        claims: List[str],
        threshold: float = 0.85,
        representative_selector: Callable[[List[str]], str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        cache_path: Optional[str] = None
) -> List[str]:
    """
    Clusters near-duplicate claims (cosine similarity >= threshold)
    then selects one representative from each cluster.

    :param claims: List of textual claims.
    :param threshold: Cosine similarity threshold for 'duplicate'.
    :param representative_selector: Function that picks 1 claim from a cluster.
                                    Defaults to picking the 'longest' claim if none provided.
    :param model_name: Embeddings model to use.
    :param device: 'cpu', 'cuda', 'mps', or None (auto).
    :param cache_path: Optional path for caching embeddings.
    :return: A list of deduplicated/representative claims.
    """
    if not claims:
        return []

    # 1) Compute embeddings
    embeddings = compute_embeddings(
        claims,
        model_name=model_name,
        device=device,
        cache_path=cache_path,
        show_progress_bar=False
    )

    # If no embeddings returned (empty claims), just return the same list
    if embeddings.size == 0:
        return claims

    # 2) Build similarity matrix
    sim_matrix = build_similarity_matrix(embeddings)

    # 3) Build adjacency from matrix
    adjacency = build_adjacency_from_matrix(sim_matrix, threshold)

    # 4) BFS to find clusters
    clusters = bfs_clusters(adjacency)

    # 5) Pick a representative for each cluster
    if representative_selector is None:
        representative_selector = select_longest  # default fallback

    deduped = []
    for cluster_indices in clusters:
        cluster_texts = [claims[idx] for idx in cluster_indices]
        rep = representative_selector(cluster_texts)
        deduped.append(rep)

    return deduped
