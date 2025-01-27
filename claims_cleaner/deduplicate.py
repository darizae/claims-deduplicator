import numpy as np
from embedding_utils import EmbeddingModel
from collections import deque
from typing import List, Callable, Optional


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
                                    Default picks the "longest" claim, for example.
    :param model_name: Embeddings model to use.
    :param device: 'cpu', 'cuda', 'mps', or None (auto).
    :param cache_path: Optional path for caching embeddings.
    :return: A list of deduplicated/representative claims.
    """
    if not claims:
        return []

    # 1) Initialize embedding model
    model = EmbeddingModel(
        model_name=model_name,
        device=device,
        cache_path=cache_path
    )

    # 2) Encode claims
    embeddings = model.encode_texts(claims, show_progress_bar=False)

    # 3) Build adjacency map (which claims are near-duplicates?)
    adjacency = {i: [] for i in range(len(claims))}
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            sim = model.compute_cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # 4) Find connected components via BFS
    visited = set()
    clusters = []
    for i in range(len(claims)):
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

    # 5) Pick a representative for each cluster
    if representative_selector is None:
        # default: pick "longest" claim
        def representative_selector(cluster_texts: List[str]) -> str:
            return max(cluster_texts, key=len)

    deduped = []
    for cluster_indices in clusters:
        cluster_texts = [claims[idx] for idx in cluster_indices]
        rep = representative_selector(cluster_texts)
        deduped.append(rep)

    return deduped
