import numpy as np
from typing import List, Dict


def build_similarity_matrix_vectorized(embeddings: np.ndarray) -> np.ndarray:
    """
    Given an (N, D) array of embeddings,
    compute the NxN cosine similarity matrix in a single step:
        sim_matrix = (E / ||E||) dot (E / ||E||)^T
    """
    # 1) Normalize embeddings along axis=1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-9  # avoid division by zero
    normed_embs = embeddings / norms

    # 2) Use matrix multiplication for pairwise dot products
    sim_matrix = normed_embs @ normed_embs.T  # shape (N, N)
    return sim_matrix


def build_adjacency_from_sim_matrix(sim_matrix: np.ndarray, threshold: float) -> Dict[int, List[int]]:
    """
    Construct adjacency from NxN sim_matrix, where sim_matrix[i,j] >= threshold => edge i<->j.
    """
    n = sim_matrix.shape[0]
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        # We only need to loop j > i to avoid duplication
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency
