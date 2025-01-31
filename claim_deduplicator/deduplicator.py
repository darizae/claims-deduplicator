from typing import List, Dict, Any, Optional, Tuple

from .embeddings import embed_unique_claims
from .clustering import cluster_claims_in_record
from .pick_representatives import pick_representatives
from .redundancy import measure_redundancy


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

    # 1) Embed unique claims
    claim2emb = embed_unique_claims(
        all_claims=claims,
        model_name=model_name,
        device=device
    )

    # 2) Since we have a single "record," just cluster them once:
    clusters = cluster_claims_in_record(
        record_claims=claims,
        claim2emb=claim2emb,
        threshold=threshold
    )

    # 3) Measure redundancy before (optional)
    redundancy_before = None
    if measure_redundancy_flag:
        redundancy_before = measure_redundancy(clusters, len(claims))

    # 4) Pick representatives
    deduped_claims = pick_representatives(claims, clusters, representative_selector)

    # 5) Optionally measure redundancy after
    redundancy_after = None
    if measure_redundancy_flag and deduped_claims:
        # cluster the deduped claims
        # Re-embed the deduped claims (or reuse claim2emb, but let's keep it consistent)
        claim2emb_deduped = embed_unique_claims(deduped_claims, model_name, device)
        clusters_after = cluster_claims_in_record(deduped_claims, claim2emb_deduped, threshold)
        redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))

    # Prepare the optional redundancy info
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

    # Gather all claims from all records to embed them once
    all_claims = []
    for rec in records:
        claims_in_rec = rec.get(field_to_deduplicate, [])
        all_claims.extend(claims_in_rec)

    # Global embedding
    claim2emb = embed_unique_claims(all_claims, model_name, device)

    clusters_output = []  # For analysis if measure_redundancy_flag is True

    for record in records:
        claims = record.get(field_to_deduplicate, [])
        if not claims:
            record[field_to_deduplicate + "_deduped"] = []
            continue

        # 1) Cluster
        clusters_before = cluster_claims_in_record(claims, claim2emb, threshold)

        # 2) Redundancy before
        red_before = None
        if measure_redundancy_flag:
            red_before = measure_redundancy(clusters_before, len(claims))

        # 3) Deduplicate (pick reps)
        deduped_claims = pick_representatives(claims, clusters_before, representative_selector)
        record[field_to_deduplicate + "_deduped"] = deduped_claims

        # 4) If measuring redundancy, cluster after dedup + measure
        if measure_redundancy_flag:
            if deduped_claims:
                # Embeddings for deduped claims
                claim2emb_deduped = {c: claim2emb[c] for c in deduped_claims}
                clusters_after = cluster_claims_in_record(deduped_claims, claim2emb_deduped, threshold)
                red_after = measure_redundancy(clusters_after, len(deduped_claims))
            else:
                clusters_after = []
                red_after = None

            # Store analysis data
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

    # 1) Gather all claims from *all* sub-lists
    all_claims = []
    for claims in list_of_claim_lists:
        all_claims.extend(claims)

    # 2) Embed them once
    claim2emb = embed_unique_claims(all_claims, model_name=model_name, device=device)

    # We'll store results in parallel arrays
    deduplicated_claim_sets = []
    analysis_info_list = []

    # 3) For each sub-list, cluster & deduplicate using the precomputed embeddings
    for i, claims in enumerate(list_of_claim_lists):
        if not claims:
            deduplicated_claim_sets.append([])
            if measure_redundancy_flag:
                analysis_info_list.append({"index": i, "redundancy_before": None, "redundancy_after": None})
            continue

        # 3a) BFS-based clustering
        clusters_before = cluster_claims_in_record(claims, claim2emb, threshold)

        # 3b) measure redundancy before
        redundancy_before = measure_redundancy(clusters_before, len(claims)) if measure_redundancy_flag else None

        # 3c) pick representatives
        deduped_claims = pick_representatives(claims, clusters_before, representative_selector)
        deduplicated_claim_sets.append(deduped_claims)

        # 3d) measure redundancy after
        redundancy_after = None
        if measure_redundancy_flag and deduped_claims:
            # We can do a second BFS on the deduplicated set
            # embedding is still the same, but let's filter claim2emb
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
