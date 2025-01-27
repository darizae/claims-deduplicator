import json
from typing import Callable, List, Optional

from .clustering import cluster_claims
from .pick_representatives import pick_representatives
from .redundancy import measure_redundancy


def deduplicate_json_file(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        representative_selector: Callable[[List[str]], str],
        threshold: float,
        model_name: str,
        device: str = None,
        separate_clusters_path: Optional[str] = None,
        measure_redundancy_flag: bool = True
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
                device=device
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
