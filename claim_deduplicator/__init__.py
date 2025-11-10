from .pipelines.deduplicator import (
    deduplicate_claims,
    deduplicate_records,
    deduplicate_multiple_claim_sets,
)
from .pipelines.file_pipeline import deduplicate_json_file
from .clustering.redundancy import measure_redundancy
from .pipelines.multi_threshold_deduplicate import multi_threshold_deduplicate

__all__ = [
    "deduplicate_claims",
    "deduplicate_records",
    "deduplicate_multiple_claim_sets",
    "deduplicate_json_file",
    "measure_redundancy",
    "multi_threshold_deduplicate",
]
