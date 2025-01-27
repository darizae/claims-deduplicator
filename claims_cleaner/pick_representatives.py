from typing import List, Callable


def pick_representatives(
        claims: List[str],
        clusters: List[List[int]],
        representative_selector: Callable[[List[str]], str]
) -> List[str]:
    """
    Given the clusters (a list of claim indices), pick exactly
    one representative from each cluster using the provided function.
    """
    deduped = []
    for cluster_indices in clusters:
        cluster_texts = [claims[idx] for idx in cluster_indices]
        rep = representative_selector(cluster_texts)
        deduped.append(rep)
    return deduped
