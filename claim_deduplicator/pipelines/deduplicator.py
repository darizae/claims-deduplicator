from typing import List, Dict, Any, Optional, Tuple

from ..embeddings.embeddings import embed_unique_claims
from ..clustering.clustering import cluster_claims_in_record
from ..clustering.pick_representatives import pick_representatives
from ..clustering.redundancy import measure_redundancy


def deduplicate_claims(
    claims: List[str],
    threshold: float = 0.85,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    measure_redundancy_flag: bool = False,
    representative_selector=None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Deduplicate a flat list of claim strings, returning a new deduplicated list.

    :param claims: A list of textual claims.
    :param threshold: Cosine similarity threshold for near-duplicates (0..1).
    :param model_name: Embedding model name for the embedding step.
    :param device: 'cpu', 'cuda', 'mps', or None for auto.
    :param measure_redundancy_flag: If True, compute redundancy metrics (before/after).
    :param representative_selector: A function that picks a representative from a cluster
                                    e.g., select_longest, select_shortest.
                                    If None, defaults to select_longest.
    :return: (deduplicated_claims, redundancy_info)
             - deduplicated_claims: list of strings, one rep per cluster
             - redundancy_info: dict with redundancy stats if measure_redundancy_flag=True,
                               otherwise an empty dict.
    """
    if representative_selector is None:
        from .strategies import select_longest
        representative_selector = select_longest

    if not claims:
        return [], {}

    claim2emb = embed_unique_claims(
        all_claims=claims,
        model_name=model_name,
        device=device
    )

    clusters = cluster_claims_in_record(
        record_claims=claims,
        claim2emb=claim2emb,
        threshold=threshold
    )

    redundancy_before = None
    if measure_redundancy_flag:
        redundancy_before = measure_redundancy(clusters, len(claims))

    deduped_claims = pick_representatives(claims, clusters, representative_selector)

    redundancy_after = None
    if measure_redundancy_flag and deduped_claims:
        claim2emb_deduped = embed_unique_claims(deduped_claims, model_name, device)
        clusters_after = cluster_claims_in_record(deduped_claims, claim2emb_deduped, threshold)
        redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))

    if measure_redundancy_flag:
        return deduped_claims, {
            "redundancy_before": redundancy_before,
            "redundancy_after": redundancy_after,
        }
    else:
        return deduped_claims, {}


def deduplicate_records(
    records: List[Dict[str, Any]],
    field_to_deduplicate: str,
    threshold: float = 0.85,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    measure_redundancy_flag: bool = False,
    representative_selector=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deduplicate claims in a list of records, each record containing a key (field_to_deduplicate)
    that is a list of textual claims.

    :param records: A list of dicts. Each dict must have a key `field_to_deduplicate`.
    :param field_to_deduplicate: The string key whose value is a list of claims in each record.
    :param threshold: Cosine similarity threshold for near-duplicates.
    :param model_name: Embedding model name.
    :param device: 'cpu', 'cuda', 'mps', or None.
    :param measure_redundancy_flag: Whether to measure redundancy stats per record.
    :param representative_selector: Strategy for picking a cluster representative.
    :return: (updated_records, analysis_data)
             - updated_records is the same input records, but each has a new key
               f"{field_to_deduplicate}_deduped" containing the deduplicated claims.
             - analysis_data is a list with redundancy info, clusters, etc.
               (one entry per record) if measure_redundancy_flag=True, else empty list.
    """
    if representative_selector is None:
        from .strategies import select_longest
        representative_selector = select_longest

    all_claims = []
    for rec in records:
        claims_in_rec = rec.get(field_to_deduplicate, [])
        all_claims.extend(claims_in_rec)

    claim2emb = embed_unique_claims(all_claims, model_name, device)

    clusters_output = []

    for record in records:
        claims = record.get(field_to_deduplicate, [])
        if not claims:
            record[field_to_deduplicate + "_deduped"] = []
            continue

        clusters_before = cluster_claims_in_record(claims, claim2emb, threshold)

        red_before = None
        if measure_redundancy_flag:
            red_before = measure_redundancy(clusters_before, len(claims))

        deduped_claims = pick_representatives(claims, clusters_before, representative_selector)
        record[field_to_deduplicate + "_deduped"] = deduped_claims

        if measure_redundancy_flag:
            if deduped_claims:
                # Embeddings for deduped claims
                claim2emb_deduped = {c: claim2emb[c] for c in deduped_claims}
                clusters_after = cluster_claims_in_record(deduped_claims, claim2emb_deduped, threshold)
                red_after = measure_redundancy(clusters_after, len(deduped_claims))
            else:
                clusters_after = []
                red_after = None

            record_id = record.get("record_id", "NO_RECORD_ID")
            clusters_output.append({
                "record_id": record_id,
                "redundancy_before": red_before,
                "redundancy_after": red_after,
                "num_claims_before": len(claims),
                "num_claims_after": len(deduped_claims),
            })

    return records, clusters_output


def deduplicate_multiple_claim_sets(
    list_of_claim_lists: List[List[str]],
    threshold: float = 0.85,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    measure_redundancy_flag: bool = False,
    representative_selector=None,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """
    Deduplicate multiple sets (lists) of claims in one go, embedding everything only once.

    :param list_of_claim_lists: A list of claim-lists. Example:
         [
           ["claim A1", "claim A2", ...],   # set #1
           ["claim B1", "claim B2", ...],   # set #2
           ...
         ]
      We'll process each set independently but embed across *all* sets in one pass.
    :param threshold: Cosine similarity threshold for near-duplicates (0..1).
    :param model_name: The sentence-transformers model to use for embeddings.
    :param device: 'cpu', 'cuda', 'mps', or None (auto).
    :param measure_redundancy_flag: If True, compute redundancy metrics before/after for each set.
    :param representative_selector: A function to pick a representative claim from each cluster
                                    (default: select_longest).
    :return: (deduplicated_claim_sets, analysis_info_list)
      - deduplicated_claim_sets: a list of deduplicated lists, parallel to `list_of_claim_lists`.
      - analysis_info_list: a list of dicts (one per set) containing optional redundancy stats
        if measure_redundancy_flag=True, else empty.
    """
    if representative_selector is None:
        from .strategies import select_longest
        representative_selector = select_longest

    all_claims = []
    for claims in list_of_claim_lists:
        all_claims.extend(claims)

    claim2emb = embed_unique_claims(all_claims, model_name=model_name, device=device)

    deduplicated_claim_sets = []
    analysis_info_list = []

    for i, claims in enumerate(list_of_claim_lists):
        if not claims:
            deduplicated_claim_sets.append([])
            if measure_redundancy_flag:
                analysis_info_list.append({"index": i, "redundancy_before": None, "redundancy_after": None})
            continue

        clusters_before = cluster_claims_in_record(claims, claim2emb, threshold)

        redundancy_before = measure_redundancy(clusters_before, len(claims)) if measure_redundancy_flag else None

        deduped_claims = pick_representatives(claims, clusters_before, representative_selector)
        deduplicated_claim_sets.append(deduped_claims)

        redundancy_after = None
        if measure_redundancy_flag and deduped_claims:
            deduped2emb = {c: claim2emb[c] for c in deduped_claims}
            clusters_after = cluster_claims_in_record(deduped_claims, deduped2emb, threshold)
            redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))

        if measure_redundancy_flag:
            analysis_info_list.append({
                "index": i,  # which set
                "num_claims_before": len(claims),
                "num_claims_after": len(deduped_claims),
                "redundancy_before": redundancy_before,
                "redundancy_after": redundancy_after,
            })

    return deduplicated_claim_sets, analysis_info_list
