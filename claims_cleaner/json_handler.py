import json
from typing import Callable, List

from .clustering import cluster_claims
from .pick_representatives import pick_representatives
from .redundancy import measure_redundancy


def deduplicate_json_file_with_redundancy(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        representative_selector: Callable[[List[str]], str],
        threshold: float,
        model_name: str,
        device: str = None,
        separate_clusters_path: str = None
):
    """
    Loads a JSON file (top-level dict of dataset->records).
    For each record:
      - measure redundancy (before)
      - cluster claims
      - pick a representative (deduplicate)
      - measure redundancy (after)
      - store results in the record:
          record[field_to_deduplicate + "_deduped"] = ...
          record["redundancy_before"] = ...
          record["redundancy_after"] = ...
      - if separate_clusters_path is provided, append cluster-level details
        (including claim texts, chosen rep, etc.) to a separate structure,
        to be saved at the end.

    Writes the updated JSON to output_json.
    Optionally writes a separate JSON with cluster details.
    """

    with open(input_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a dict at top level (e.g., cnndm_test, etc.)")

    clusters_output = []  # for storing cluster analysis if separate_clusters_path is not None

    for dataset_name, records in data.items():
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of records for dataset {dataset_name}.")

        for record in records:
            claims = record.get(field_to_deduplicate, [])
            if not isinstance(claims, list):
                raise ValueError(f"Field '{field_to_deduplicate}' is not a list in record {record}.")

            # 1. cluster once
            clusters = cluster_claims(
                claims=claims,
                threshold=threshold,
                model_name=model_name,
                device=device
            )

            # measure redundancy BEFORE (based on the original number of claims)
            redundancy_before = measure_redundancy(clusters, len(claims))

            # 2. pick representatives
            deduped_claims = pick_representatives(claims, clusters, representative_selector)

            # measure redundancy AFTER
            # we can just do a fresh cluster on deduped_claims:
            clusters_after = cluster_claims(
                claims=deduped_claims,
                threshold=threshold,
                model_name=model_name,
                device=device
            )
            redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))

            # 3. Store results in record
            record[field_to_deduplicate + "_deduped"] = deduped_claims
            record["redundancy_before"] = redundancy_before
            record["redundancy_after"] = redundancy_after

            # 4. If we want a separate clusters file, store cluster details
            if separate_clusters_path is not None:
                # gather cluster info for each cluster
                # example structure
                record_id = record.get("record_id", "NO_ID")

                cluster_details = []
                for c_index, c_indices in enumerate(clusters):
                    cluster_texts = [claims[i] for i in c_indices]
                    rep_claim = deduped_claims[clusters.index(c_indices)] if len(deduped_claims) > 0 else None
                    # or do: rep_claim = representative_selector(cluster_texts)
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
                    "redundancy_before": redundancy_before,
                    "deduplicated_claims": deduped_claims,
                    "redundancy_after": redundancy_after
                })

    # 5. Write updated data with deduplicated claims and redundancy stats
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 6. If separate_clusters_path is specified, write out the cluster-level details
    if separate_clusters_path is not None:
        with open(separate_clusters_path, "w") as f:
            json.dump(clusters_output, f, indent=2)
