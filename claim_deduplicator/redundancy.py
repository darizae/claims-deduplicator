from typing import List, Dict


def measure_redundancy(
        clusters: List[List[int]],
        total_claims: int
) -> Dict[str, float]:
    """
    Given clusters (a list of lists of indices) and the total number of claims,
    compute redundancy metrics.

    Returns a dictionary with:
        - 'num_clusters'
        - 'redundancy_ratio' (avg cluster size)
        - 'fraction_duplicates' (0..1)
        - 'unique_claims_pct' (0..1)
    """
    if total_claims == 0:
        return {
            "num_clusters": 0,
            "redundancy_ratio": 1.0,
            "fraction_duplicates": 0.0,
            "unique_claims_pct": 0.0
        }

    num_clusters = len(clusters)
    redundancy_ratio = total_claims / float(num_clusters)
    fraction_duplicates = (total_claims - num_clusters) / float(total_claims)
    unique_claims_pct = num_clusters / float(total_claims)

    return {
        "num_clusters": num_clusters,
        "redundancy_ratio": redundancy_ratio,
        "fraction_duplicates": fraction_duplicates,
        "unique_claims_pct": unique_claims_pct
    }
