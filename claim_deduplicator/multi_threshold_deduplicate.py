import json
import os
from typing import List, Dict, Callable, Optional
import numpy as np

# Reuse your existing logic
from .embeddings import compute_embeddings
from .scoring import build_similarity_matrix_vectorized, build_adjacency_from_sim_matrix
from .clustering import bfs_clusters
from .pick_representatives import pick_representatives
from .redundancy import measure_redundancy


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def multi_threshold_deduplicate(
        input_json_path: str,
        output_json_path: str,
        thresholds: List[float],
        representative_selector: Callable[[List[str]], str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        measure_redundancy_flag: bool = False,
        cluster_analysis_dir: Optional[str] = None,
        embeddings_cache_path: Optional[str] = None
):
    """
    Deduplicate the 'reference_acus["original"]' field for each record at multiple thresholds,
    storing the results under 'reference_acus["deduped_{threshold}_{strategy}"]'.

    The NxN similarity matrix is built once per record and reused for all thresholds,
    which is more efficient than recomputing NxN each time.

    Steps:
    1) Load the new-schema JSON (with reference_acus.original).
    2) Collect and embed all original claims once (global).
    3) For each record:
        - Gather the embeddings for that record's original claims, build NxN similarity.
        - For each threshold: build adjacency, BFS, pick reps, store in 'deduped_{threshold}_{strategy}'.
          If measure_redundancy_flag=True, also do BFS on deduped claims & log cluster analysis.
    4) Save one final JSON with all deduped sets.

    :param input_json_path: Path to the *migrated* RoSE dataset JSON (new schema).
    :param output_json_path: Where to save the updated JSON.
    :param thresholds: A list of float thresholds, e.g. [0.1, 0.2, ..., 0.9].
    :param representative_selector: e.g. `select_longest`, `select_shortest`, ...
    :param model_name: The SBERT or embedding model to use.
    :param device: 'cpu', 'cuda', 'mps', or None (auto-detect).
    :param measure_redundancy_flag: Whether to measure redundancy for each threshold.
    :param cluster_analysis_dir: If provided and measure_redundancy_flag=True,
        we'll write separate cluster_analysis_{threshold}.json for each threshold.
    :param embeddings_cache_path: If you want to specify a custom path for cached embeddings.
    """

    # 1) Load data
    data = load_json(input_json_path)

    # Gather all "original" claims from the entire dataset
    # so we can embed them only once
    all_original_claims = []
    for subset_name, records in data.items():
        for record in records:
            ref_acus = record.get("reference_acus", {})
            original_claims = ref_acus.get("original", [])
            all_original_claims.extend(original_claims)

    # Remove duplicates globally
    unique_claims = list(set(all_original_claims))
    print(f"[multi_threshold_deduplicate] #unique original claims: {len(unique_claims)}")

    # 2) Embed them once
    all_embeddings = compute_embeddings(
        unique_claims,
        model_name=model_name,
        device=device,
        cache_path=embeddings_cache_path,
        show_progress_bar=True
    )

    # Build a map text->embedding
    claim2emb = {}
    for txt, emb in zip(unique_claims, all_embeddings):
        claim2emb[txt] = emb

    # Prepare cluster analysis logs if measuring redundancy
    # We'll accumulate logs in a dict keyed by threshold, then write them out
    cluster_analysis_logs = {th: [] for th in thresholds}
    strategy_name = representative_selector.__name__

    # 3) For each record, build NxN once and reuse for all thresholds
    for subset_name, records in data.items():
        for record in records:
            ref_acus = record.setdefault("reference_acus", {})
            original_claims = ref_acus.get("original", [])

            if not original_claims:
                # If no claims, skip
                for th in thresholds:
                    # Just store empty deduped array
                    th_str = f"{th:.2f}".rstrip("0").rstrip(".")
                    ref_acus[f"deduped_{th_str}_{strategy_name}"] = []
                continue

            # a) Gather embeddings for these original claims
            embs = np.array([claim2emb[txt] for txt in original_claims])

            # b) Build NxN sim matrix once
            sim_matrix = build_similarity_matrix_vectorized(embs)

            # c) For each threshold, build adjacency, BFS, pick reps, measure redundancy if needed
            for th in thresholds:
                th_str = f"{th:.2f}".rstrip("0").rstrip(".")
                # Build adjacency
                adjacency = build_adjacency_from_sim_matrix(sim_matrix, th)
                clusters_before = bfs_clusters(adjacency)

                deduped_claims = pick_representatives(original_claims, clusters_before, representative_selector)
                ref_acus[f"deduped_{th_str}_{strategy_name}"] = deduped_claims

                if measure_redundancy_flag:
                    # measure redundancy before
                    redundancy_before = measure_redundancy(clusters_before, len(original_claims))

                    # BFS on deduped to measure after
                    if deduped_claims:
                        d_embs = np.array([claim2emb[txt] for txt in deduped_claims])
                        d_sim = build_similarity_matrix_vectorized(d_embs)
                        d_adj = build_adjacency_from_sim_matrix(d_sim, th)
                        clusters_after = bfs_clusters(d_adj)
                        redundancy_after = measure_redundancy(clusters_after, len(deduped_claims))
                    else:
                        clusters_after = []
                        redundancy_after = None

                    # prepare cluster details
                    cluster_details = []
                    for c_index, c_indices in enumerate(clusters_before):
                        cluster_texts = [original_claims[i] for i in c_indices]
                        rep_claim = deduped_claims[c_index]
                        cluster_details.append({
                            "cluster_id": c_index,
                            "cluster_size": len(c_indices),
                            "cluster_texts": cluster_texts,
                            "representative_claim": rep_claim
                        })

                    record_id = record.get("record_id", "NO_ID")
                    cluster_analysis_logs[th].append({
                        "dataset_name": subset_name,
                        "record_id": record_id,
                        "threshold": th,
                        "strategy": strategy_name,
                        "clusters": cluster_details,
                        "redundancy_before": redundancy_before,
                        "redundancy_after": redundancy_after
                    })

    # 4) Write final JSON
    write_json(data, output_json_path)
    print(f"[multi_threshold_deduplicate] Wrote updated data with deduped claims to {output_json_path}")

    # 5) Write cluster logs if measure_redundancy_flag == True
    if measure_redundancy_flag and cluster_analysis_dir:
        os.makedirs(cluster_analysis_dir, exist_ok=True)
        for th, logs in cluster_analysis_logs.items():
            th_str = f"{th:.2f}".rstrip("0").rstrip(".")
            out_path = os.path.join(cluster_analysis_dir, f"cluster_analysis_{th_str}_{strategy_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"Saved cluster analysis for threshold={th_str} to {out_path}")
