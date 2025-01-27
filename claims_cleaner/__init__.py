# claims_cleaner/__init__.py

from .deduplicate import deduplicate_claims
from .json_handler import deduplicate_json_file
from .strategies import select_longest, select_shortest, select_random
from .clustering import bfs_clusters
from .scoring import (
    compute_embeddings,
    compute_cosine_similarity,
    build_similarity_matrix,
    build_adjacency_from_matrix
)

__all__ = [
    "deduplicate_claims",
    "deduplicate_json_file",
    "select_longest",
    "select_shortest",
    "select_random",
    "bfs_clusters",
    "compute_embeddings",
    "compute_cosine_similarity",
    "build_similarity_matrix",
    "build_adjacency_from_matrix",
]
