from claim_deduplicator import deduplicate_claims

if __name__ == "__main__":
    claims = [
        "Jane is in Warsaw",
        "Ollie has a party",
        "Jane has a party",
        "Jane lost her calendar",
        "Ollie and Jane will get a lunch",
        "Ollie and Jane will get a lunch this week",
        "Ollie and Jane will get a lunch on Friday",
        "Ollie accidentally called Jane",
        "Ollie talked about whisky",
        "Jane cancels lunch",
        "Ollie and Jane will meet for a tea",
        "Ollie and Jane will meet for a tea at 6 pm"
    ]

    deduped, stats = deduplicate_claims(
        claims,
        threshold=0.85,
        measure_redundancy_flag=True
    )
    print("Deduplicated claims:", deduped)
    print("Redundancy stats:", stats)