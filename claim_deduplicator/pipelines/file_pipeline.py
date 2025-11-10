import json
from typing import Optional, Callable

from .deduplicator import deduplicate_records


def deduplicate_json_file(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        threshold: float = 0.85,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        measure_redundancy_flag: bool = True,
        representative_selector: Optional[Callable] = None,
        cluster_analysis_path: Optional[str] = None
) -> None:
    """
    Orchestrate the entire pipeline reading from a JSON file, deduplicating,
    then writing results back to disk.
    Expects the JSON shape: { "dataset_name": [ {record}, {record}, ... ], ... }.

    :param input_json_path: Path to the input JSON.
    :param output_json_path: Where to save the deduplicated JSON.
    :param field_to_deduplicate: Key in each record whose value is a list of claims.
    :param threshold: Cosine similarity threshold.
    :param model_name: e.g. "sentence-transformers/all-MiniLM-L6-v2".
    :param device: 'cpu', 'cuda', 'mps', or None.
    :param measure_redundancy_flag: If True, compute redundancy stats.
    :param representative_selector: Strategy for picking the representative claim in each cluster.
    :param cluster_analysis_path: If provided, write cluster analysis as a separate JSON file.
    """
    # 1) Load
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data is expected to be { dataset_name -> list of records }
    analysis_output = []

    for dataset_name, records in data.items():
        updated_records, analysis_data = deduplicate_records(
            records=records,
            field_to_deduplicate=field_to_deduplicate,
            threshold=threshold,
            model_name=model_name,
            device=device,
            measure_redundancy_flag=measure_redundancy_flag,
            representative_selector=representative_selector,
        )
        # store them back in data
        data[dataset_name] = updated_records

        # optional analysis
        if measure_redundancy_flag:
            for entry in analysis_data:
                entry["dataset_name"] = dataset_name
            analysis_output.extend(analysis_data)

    # 2) Write updated JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 3) If measure redundancy and path is given, write cluster analysis
    if measure_redundancy_flag and cluster_analysis_path:
        with open(cluster_analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_output, f, indent=2)
