import json
from typing import Callable, List


def deduplicate_json_file(
        input_json_path: str,
        output_json_path: str,
        field_to_deduplicate: str,
        deduplicate_fn: Callable[[List[str]], List[str]],
):
    """
    Loads a JSON file (list of records), for each record:
    - Takes the array from 'field_to_deduplicate'
    - Runs 'deduplicate_fn' on it
    - Stores the result in e.g. 'field_to_deduplicate + "_deduped"'
    Saves the updated records to output JSON.
    """
    with open(input_json_path, "r") as f:
        records = json.load(f)

    if not isinstance(records, dict):
        raise ValueError(f"Expected a dictionary at the top level, got {type(records)}")

    cleaned_data = {}

    # Iterate through each dataset (e.g., "cnndm_test", "cnndm_validation", etc.)
    for dataset_name, dataset_records in records.items():
        if not isinstance(dataset_records, list):
            raise ValueError(f"Expected list for dataset {dataset_name}, got {type(dataset_records)}")

        cleaned_records = []
        for record in dataset_records:
            if not isinstance(record, dict):
                raise ValueError(f"Expected dictionary, got {type(record)}: {record}")

            # Deduplicate the field
            claims = record.get(field_to_deduplicate, [])
            if not isinstance(claims, list):
                raise ValueError(f"Expected a list in field '{field_to_deduplicate}', got {type(claims)}")

            deduplicated_claims = deduplicate_fn(claims)
            record[field_to_deduplicate + "_deduped"] = deduplicated_claims
            cleaned_records.append(record)

        cleaned_data[dataset_name] = cleaned_records  # Store cleaned data

    with open(output_json_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)
