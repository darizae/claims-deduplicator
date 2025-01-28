import numpy as np

from .clustering import cluster_claims, cluster_claims_in_record
from .pick_representatives import pick_representatives
from .redundancy import measure_redundancy
from claims_cleaner.scoring import compute_embeddings

import json
from typing import Dict, List, Callable, Optional


def opt_deduplicate_json_file(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        representative_selector: Callable[[List[str]], str],
        threshold: float,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        measure_redundancy_flag: bool = True,
):
    """
    - Load the entire JSON.
    - Gather all claims across records, embed them once -> claim2emb.
    - For each record, do BFS-based clustering in local scope (fast, no re-embedding).
    - Optionally do BFS after for deduplicated claims.

    Writes out deduplicated claims to output, can also measure redundancy if flagged.
    """
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 1) Gather all claims across datasets
    all_claims = gather_all_claims(data, field_to_deduplicate=field_to_deduplicate)

    # 2) Embed them once
    claim2emb = embed_unique_claims(
        all_claims,
        model_name=model_name,
        device=device
    )

    # 3) Initialize clusters output
    clusters_output = []

    for dataset_name, records in data.items():
        for record in records:
            record_claims = record.get(field_to_deduplicate, [])
            if not record_claims:
                record[field_to_deduplicate + "_deduped"] = []
                continue

            # 4) Cluster **before** deduplication
            clusters_before = cluster_claims_in_record(record_claims, claim2emb, threshold)

            # 5) Measure redundancy **before deduplication**
            redundancy_before = measure_redundancy(clusters_before, len(record_claims)) if measure_redundancy_flag else None

            # 6) Deduplicate: Pick **one representative** per cluster
            deduped_claims = pick_representatives(record_claims, clusters_before, representative_selector)
            record[field_to_deduplicate + "_deduped"] = deduped_claims

            # 7) Re-use adjacency list to compute **redundancy after**
            clusters_after = cluster_claims_in_record(deduped_claims, claim2emb, threshold) if measure_redundancy_flag else []
            redundancy_after = measure_redundancy(clusters_after, len(deduped_claims)) if measure_redundancy_flag else None

            # 8) Format output of cluster analysis
            if measure_redundancy_flag:
                record_id = record.get("record_id", "NO_ID_PROVIDED")

                cluster_details = []
                for c_index, c_indices in enumerate(clusters_before):
                    cluster_texts = [record_claims[i] for i in c_indices]
                    rep_claim = deduped_claims[c_index]  # because each cluster -> 1 rep
                    cluster_details.append({
                        "cluster_id": c_index,
                        "cluster_size": len(c_indices),
                        "cluster_texts": cluster_texts,
                        "representative_claim": rep_claim
                    })

                clusters_output.append({
                    "dataset_name": dataset_name,
                    "record_id": record_id,
                    "clusters": cluster_details,
                    "deduplicated_claims": deduped_claims,
                    "redundancy_before": redundancy_before,
                    "redundancy_after": redundancy_after
                })

    # 9) Write deduplicated results
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 10) Write cluster analysis
    if measure_redundancy_flag:
        with open("data/cluster_analysis.json", "w") as f:
            json.dump(clusters_output, f, indent=2)


def deduplicate_json_file(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        representative_selector: Callable[[List[str]], str],
        threshold: float,
        model_name: str,
        device: str = None,
        separate_clusters_path: Optional[str] = None,
        measure_redundancy_flag: bool = True,
        cache_path: Optional[str] = None
):
    """
    Deduplicate a JSON dataset by clustering near-duplicate claims and picking one representative
    for each cluster. Optionally measure redundancy metrics and store them in a separate clusters file.

    **Expected Input JSON Structure**:
      {
        "dataset_name_1": [
          {
            "record_id": "some_unique_id_0",
            "source": "...",
            "reference_acus": ["Claim 1...", "Claim 2..."],
            ...
          },
          {
            "record_id": "some_unique_id_1",
            ...
          }
        ],
        "dataset_name_2": [...],
        ...
      }

    - The top-level is a dict whose keys are dataset names (e.g. 'cnndm_test').
    - Each dataset maps to a list of records.
    - Each record is a dict that may contain the field you want to deduplicate.

    :param input_json_path: Path to the input JSON file.
    :param output_json_path: Path to write the deduplicated JSON output.
    :param field_to_deduplicate: Name of the key in each record whose value is a list of claims.
    :param representative_selector: A callable that, given a list of claims in a cluster, returns one representative.
    :param threshold: Cosine similarity threshold for near-duplicates (0..1).
    :param model_name: Name of the embedding model (HuggingFace or SBERT).
    :param device: 'cpu', 'cuda', 'mps', or None (auto-detect).
    :param separate_clusters_path: If provided, cluster-level details are written to this separate JSON.
    :param measure_redundancy_flag: Whether to measure redundancy metrics. If False, skip that step
                                    (useful to save computation time).
    :param cache_path:
    :return: None (writes JSON to disk).

    **Notes**:
    - The main output JSON (RoSE JSON) will contain the same structure as the input, but each record
      will have an additional field: `<field_to_deduplicate>_deduped` with the final claims list.
    - Redundancy metrics (before/after) are NOT stored in the main output if measure_redundancy_flag=False.
      Even if True, we only write them to the separate clusters file (if given).
    """

    with open(input_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a dict at top-level: {dataset_name: [records]}")

    # Will hold cluster-level info (if separate_clusters_path is not None):
    clusters_output = []

    for dataset_name, records in data.items():
        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list of records under dataset '{dataset_name}', got {type(records)}"
            )

        for record in records:
            claims = record.get(field_to_deduplicate, [])
            if not isinstance(claims, list):
                raise ValueError(
                    f"Field '{field_to_deduplicate}' must be a list, got {type(claims)} in record {record}"
                )

            # 1) Cluster once
            clusters = cluster_claims(
                claims=claims,
                threshold=threshold,
                model_name=model_name,
                device=device,
                cache_path=cache_path
            )

            # 2) Pick representatives -> deduplicated claims
            deduped_claims = pick_representatives(claims, clusters, representative_selector)
            record[field_to_deduplicate + "_deduped"] = deduped_claims

            # 3) If separate_clusters_path, gather cluster analysis
            if separate_clusters_path is not None:
                record_id = record.get("record_id", "NO_ID_PROVIDED")
                cluster_details = []
                for c_index, c_indices in enumerate(clusters):
                    cluster_texts = [claims[i] for i in c_indices]
                    rep_claim = deduped_claims[c_index]  # because each cluster -> 1 rep
                    cluster_details.append({
                        "cluster_id": c_index,
                        "cluster_size": len(c_indices),
                        "cluster_texts": cluster_texts,
                        "representative_claim": rep_claim
                    })

                clusters_dict = {
                    "dataset_name": dataset_name,
                    "record_id": record_id,
                    "clusters": cluster_details,
                    "deduplicated_claims": deduped_claims,
                }

                # 4) Measure redundancy only if flagged
                if measure_redundancy_flag:
                    redundancy_before = measure_redundancy(clusters, len(claims))
                    # Re-cluster the deduped set just for measuring after?
                    # Or skip it if you want to reduce overhead.
                    clusters_after = cluster_claims(
                        deduped_claims, threshold=threshold, model_name=model_name, device=device
                    )
                    redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))
                    clusters_dict["redundancy_before"] = redundancy_before
                    clusters_dict["redundancy_after"] = redundancy_after

                clusters_output.append(clusters_dict)

    # Write the main (RoSE) JSON with deduplicated claims
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Write the separate clusters JSON only if requested
    if separate_clusters_path is not None:
        with open(separate_clusters_path, "w") as f:
            json.dump(clusters_output, f, indent=2)


def gather_all_claims(data: Dict[str, List[Dict]], field_to_deduplicate: str) -> List[str]:
    """
    Collects all claims from the given JSON structure.
    data = {
      "dataset_name": [ {record}, {record}, ... ],
      ...
    }
    Returns a list of (possibly duplicated) claim strings.
    """
    all_claims = []
    for dataset_name, records in data.items():
        for record in records:
            claims = record.get(field_to_deduplicate, [])
            all_claims.extend(claims)
    return all_claims


def embed_unique_claims(
    all_claims: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = None
) -> Dict[str, np.ndarray]:
    """
    Given a big list of claims (with duplicates),
    1) Convert to a set of unique claims,
    2) Embed them in one shot,
    3) Return a dict {claim_text: embedding_vector}.
    """

    unique_claims = list(set(all_claims))  # deduplicate text
    print(f"Total claims: {len(all_claims)}; unique: {len(unique_claims)}")

    # EMBED everything in a single batch
    all_embeddings = compute_embeddings(
        unique_claims,
        model_name=model_name,
        device=device,
        show_progress_bar=True
    )

    # Build the dictionary
    claim2emb = {}
    for text, emb in zip(unique_claims, all_embeddings):
        claim2emb[text] = emb

    return claim2emb
