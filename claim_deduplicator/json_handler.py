import json
from typing import Dict, List, Callable, Optional

from .clustering import cluster_claims_in_record
from .pick_representatives import pick_representatives
from .embeddings import embed_unique_claims
from .redundancy import measure_redundancy


def load_json(input_path: str) -> Dict:
    """Load JSON from the specified file path."""
    with open(input_path, "r") as f:
        return json.load(f)


def write_json(data: Dict, output_path: str) -> None:
    """Write JSON to the specified file path."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_cluster_analysis(clusters_output: List[Dict], analysis_path: str) -> None:
    """Write cluster analysis JSON to a dedicated file."""
    with open(analysis_path, "w") as f:
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


def deduplicate_data(
    data: Dict[str, List[Dict]],
    field_to_deduplicate: str,
    representative_selector: Callable[[List[str]], str],
    threshold: float,
    claim2emb: Dict[str, 'np.ndarray'],  # or just np.ndarray if you prefer
    measure_redundancy_flag: bool = True
) -> (Dict[str, List[Dict]], List[Dict]):
    """
    Core logic that:
      1) Iterates over all datasets and records in 'data'.
      2) Clusters the claims, picks one representative per cluster.
      3) (Optional) measures redundancy before/after deduplication.
      4) Appends cluster info to a 'clusters_output' list for analysis.

    Returns:
      - Modified 'data' with a new key "<field_to_deduplicate>_deduped"
        containing the deduplicated claims in each record.
      - 'clusters_output' for any further analysis or writing to disk.
    """
    clusters_output = []

    for dataset_name, records in data.items():
        for record in records:
            record_claims = record.get(field_to_deduplicate, [])
            if not record_claims:
                record[field_to_deduplicate + "_deduped"] = []
                continue

            # Cluster claims in the current record
            clusters_before = cluster_claims_in_record(record_claims, claim2emb, threshold)

            # Measure redundancy before dedup
            redundancy_before = measure_redundancy(clusters_before, len(record_claims)) \
                if measure_redundancy_flag else None

            # Deduplicate: pick one representative per cluster
            deduped_claims = pick_representatives(record_claims, clusters_before, representative_selector)
            record[field_to_deduplicate + "_deduped"] = deduped_claims

            # If measuring redundancy, cluster the deduplicated version
            if measure_redundancy_flag:
                clusters_after = cluster_claims_in_record(deduped_claims, claim2emb, threshold)
                redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))
            else:
                clusters_after = []
                redundancy_after = None

            if measure_redundancy_flag:
                record_id = record.get("record_id", "NO_ID_PROVIDED")
                cluster_details = []
                for c_index, c_indices in enumerate(clusters_before):
                    cluster_texts = [record_claims[i] for i in c_indices]
                    # deduped_claims is in the same order as clusters_before,
                    # i.e. each cluster corresponds to one representative
                    rep_claim = deduped_claims[c_index]
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

    return data, clusters_output


def deduplicate_json_file(
    input_json_path: str,
    output_json_path: str,
    field_to_deduplicate: str,
    representative_selector: Callable[[List[str]], str],
    threshold: float,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    measure_redundancy_flag: bool = True,
    cluster_analysis_path: str = "data/cluster_analysis.json"
) -> None:
    """
    Orchestrates the entire pipeline:
      1) Loads JSON from disk.
      2) Gathers all claims and embeds them once.
      3) Deduplicates each record in memory.
      4) Writes the updated data to disk.
      5) Optionally writes a separate cluster analysis JSON.

    :param input_json_path: Path to the input JSON file.
    :param output_json_path: Path to the output JSON file.
    :param field_to_deduplicate: Name of the key in each record whose value is a list of claims.
    :param representative_selector: A callable that, given a list of claims in a cluster,
                                    returns one representative.
    :param threshold: Cosine similarity threshold for near-duplicates (0..1).
    :param model_name: Embedding model name for the embedding step.
    :param device: 'cpu', 'cuda', 'mps', or None (auto-detect).
    :param measure_redundancy_flag: Whether to measure redundancy metrics.
    :param cluster_analysis_path: Where to write the cluster analysis JSON (if measuring redundancy).
    """
    # 1) Load data from disk
    data = load_json(input_json_path)

    # 2) Gather and embed all claims
    all_claims = gather_all_claims(data, field_to_deduplicate)
    claim2emb = embed_unique_claims(
        all_claims,
        model_name=model_name,
        device=device
    )

    # 3) Deduplicate in-memory
    updated_data, clusters_output = deduplicate_data(
        data=data,
        field_to_deduplicate=field_to_deduplicate,
        representative_selector=representative_selector,
        threshold=threshold,
        claim2emb=claim2emb,
        measure_redundancy_flag=measure_redundancy_flag
    )

    # 4) Write updated JSON
    write_json(updated_data, output_json_path)

    # 5) If redundancy measurement was requested, write cluster analysis
    if measure_redundancy_flag:
        write_cluster_analysis(clusters_output, cluster_analysis_path)
