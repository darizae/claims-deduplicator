from .deduplicate import deduplicate_claims
from .json_handler import deduplicate_json_file
from .strategies import select_longest, select_shortest, select_random

__all__ = [
    "deduplicate_claims",
    "deduplicate_json_file",
    "select_longest",
    "select_shortest",
    "select_random",
]
