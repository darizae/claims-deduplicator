import argparse

from claim_deduplicator.json_handler import deduplicate_json_file
from .strategies import select_longest, select_shortest, select_random

STRATEGY_MAP = {
    "longest": select_longest,
    "shortest": select_shortest,
    "random": select_random
}


def main():
    parser = argparse.ArgumentParser(
        description="CLI for claim-deduplicator: deduplicates claims in a JSON file."
    )
    parser.add_argument("--input-json", required=True,
                        help="Path to input JSON file.")
    parser.add_argument("--output-json", required=True,
                        help="Path to output JSON file.")
    parser.add_argument("--field-to-deduplicate", required=True,
                        help="Field name (list of claims) to deduplicate.")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for near-duplicates.")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace/SBERT model name to use for embeddings.")
    parser.add_argument("--strategy", choices=STRATEGY_MAP.keys(), default="longest",
                        help="Strategy for picking a representative claim from each cluster.")
    parser.add_argument("--device", default=None,
                        help="Device to run embeddings on: 'cpu', 'cuda', 'mps', or None (auto).")
    parser.add_argument("--measure-redundancy", action="store_true",
                        help="If set, compute redundancy metrics and store them in the clusters JSON.")
    parser.add_argument("--cache-path", default=None,
                        help="If provided, embeddings will be cached to this file. Otherwise, a default path is used.")

    args = parser.parse_args()

    deduplicate_json_file(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        field_to_deduplicate=args.field_to_deduplicate,
        representative_selector=STRATEGY_MAP[args.strategy],
        threshold=args.threshold,
        model_name=args.model_name,
        device=args.device,
        measure_redundancy_flag=args.measure_redundancy,
    )


if __name__ == "__main__":
    main()
