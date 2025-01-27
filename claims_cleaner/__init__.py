from .deduplicate import deduplicate_claims
from .json_handler import deduplicate_json_file
from .redundancy import measure_redundancy
from .strategies import select_longest, select_shortest, select_random
from .clustering import bfs_clusters, cluster_claims
from .scoring import (
    compute_embeddings,
    compute_cosine_similarity,
    build_similarity_matrix,
    build_adjacency_from_matrix
)

__all__ = [
    "cluster_claims",
    "pick_representatives",
    "deduplicate_claims",
    "deduplicate_json_file",
    "select_longest",
    "select_shortest",
    "select_random",
    "bfs_clusters",
    "compute_embeddings",
    "measure_redundancy",
    "compute_cosine_similarity",
    "build_similarity_matrix",
    "build_adjacency_from_matrix",
]
