from typing import List, Callable, Optional

from .clustering import cluster_claims
from .pick_representatives import pick_representatives
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
    Clusters near-duplicate claims, picks 1 representative from each cluster,
    and returns the deduplicated list of claims.
    """
    if not claims:
        return []

    clusters = cluster_claims(
        claims=claims,
        threshold=threshold,
        model_name=model_name,
        device=device,
        cache_path=cache_path,
    )

    if representative_selector is None:
        representative_selector = select_longest

    deduped = pick_representatives(claims, clusters, representative_selector)
    return deduped
