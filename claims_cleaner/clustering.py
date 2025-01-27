from collections import deque
from typing import Dict, List


def bfs_clusters(adjacency: Dict[int, List[int]]) -> List[List[int]]:
    """
    Perform BFS to find connected components in an adjacency list.

    :param adjacency: A dict {i: [neighbors...], ...}.
    :return: A list of clusters, where each cluster is a list of indices.
    """
    visited = set()
    clusters = []

    for i in adjacency:
        if i not in visited:
            queue = deque([i])
            cluster = []
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            clusters.append(cluster)

    return clusters
