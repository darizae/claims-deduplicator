from claims_cleaner.deduplicate import deduplicate_claims
from claims_cleaner.json_handler import deduplicate_json_file


def main():
    deduplicate_json_file(
        input_json_path="my_data.json",
        output_json_path="my_data_deduped.json",
        field_to_deduplicate="reference_acus",
        deduplicate_fn=lambda claims: deduplicate_claims(
            claims,
            threshold=0.85
        )
    )


if __name__ == "__main__":
    main()
