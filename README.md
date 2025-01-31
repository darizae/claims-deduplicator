# claims-deduplicator

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Project Status - Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)](#)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

This repository provides a Python package and CLI tool for **deduplicating textual claims** via sentence embeddings, BFS-based clustering, and representative selection.
  

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
   - [Using as a Python Library](#using-as-a-python-library)
   - [Using the CLI](#using-the-cli)
4. [Example Code Snippet](#example-code-snippet)
5. [Under the Hood: Vectorized Cosine Similarity](#under-the-hood-vectorized-cosine-similarity)
6. [Multi-Threshold Deduplication](#multi-threshold-deduplication)
7. [License](#license)

---

## Key Features
- **Sentence Embeddings** with caching, using `sentence-transformers` and [`embedding-utils`](https://pypi.org/project/embedding-utils/)  
- **BFS Clustering** to group claims with cosine similarity above a threshold  
- **Representative Selection** (e.g. longest, shortest, random) from each cluster  
- **Redundancy Metrics** like fraction of duplicates, average cluster size, etc.  
- **File-based Pipeline** for JSON data (deduplicates your dataset’s claim fields in-place)  
- **Multi-Threshold** approach to generate multiple deduplications in a single run  

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/claims-deduplicator.git
   cd claims-deduplicator
   ```

2. **Install with pip** (into a virtual environment or conda environment of your choice):

   ```bash
   pip install .
   ```
   <br/>
3.  Or, if you prefer **Conda** and want the exact pinned packages, use:
   ```bash
   conda env create -f environment.yml
   conda activate claimDedup
   ```
   *Ensure your environment is active before running any commands.*

---

## Quick Start

### Using as a Python Library

After installation, you can import the main functions directly:

```python
from claim_deduplicator import deduplicate_claims

claims = [
    "Jane is in Warsaw",
    "Ollie has a party",
    "Jane has a party",
    "Jane lost her calendar"
]

deduped_claims, stats = deduplicate_claims(
    claims,
    threshold=0.85,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    measure_redundancy_flag=True
)

print("Deduplicated claims:", deduped_claims)
print("Stats:", stats)
```

**What’s happening?**  
- Unique claim strings get embedded via a Sentence Transformers model  
- A similarity matrix is built, and BFS clustering groups near-duplicates above `threshold=0.85`  
- A single representative from each cluster is chosen (defaults to the longest claim)  
- If `measure_redundancy_flag=True`, you’ll also see metrics like `fraction_duplicates`, `unique_claims_pct`, etc.

### Using the CLI

The CLI entry point `claims-deduplicator` (installed automatically) reads and writes JSON files. 
Suppose you have `data/example_dataset.json` with a structure like:

```json
{
  "my_dataset": [
    {
      "source": "...",
      "reference_acus": ["claim 1", "claim 2", ...]
    },
    ...
  ]
}
```

You can deduplicate the `reference_acus` lists in-place with:

```bash
claims-deduplicator \
  --input-json data/example_dataset.json \
  --output-json data/example_dataset_deduped.json \
  --field-to-deduplicate reference_acus \
  --threshold 0.85 \
  --measure-redundancy
```

Now each record will get a new key named `reference_acus_deduped` containing deduplicated claims.

Below is a quick overview of the CLI arguments supported.

### CLI Parameters

| Parameter               | Description                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `--input-json`          | Path to the input JSON file.                                                                             |
| `--output-json`         | Path to the output JSON file with deduplicated results.                                                  |
| `--field-to-deduplicate`| Field name (within each record) containing the list of textual claims.                                   |
| `--threshold`           | Cosine similarity threshold (0–1). Claims above this similarity are considered duplicates.               |
| `--model-name`          | Model name for sentence embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`).                     |
| `--device`              | Compute device: `"cpu"`, `"cuda"`, `"mps"`, or None (auto-detect).                                       |
| `--strategy`            | Representative selection strategy: `"longest"`, `"shortest"`, `"random"`.                                |
| `--measure-redundancy`  | If set, computes redundancy metrics (before/after).                                                      |

---

## Example Code Snippet

To see a real minimal usage example, check out [`example_deduplication.py`](./example_deduplication.py). It demonstrates calling `deduplicate_claims(...)` on an inline list of claims.

```bash
python example_deduplication.py
```

Which prints something like:
```
Deduplicated claims: [...]
Redundancy stats: {...}
```

---

Here is the text formatted in Markdown, ready to be copied and pasted into a README file:

## Under the Hood: Vectorized Cosine Similarity

We **vectorize** the entire process:

```python
def build_similarity_matrix_vectorized(embeddings: np.ndarray) -> np.ndarray:
    """
    Given an (N, D) array of embeddings,
    compute the NxN cosine similarity matrix in a single step:
        sim_matrix = (E / ||E||) dot (E / ||E||)^T
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-9  
    normed_embs = embeddings / norms

    sim_matrix = normed_embs @ normed_embs.T
    return sim_matrix
```

### Mathematical Basis

For two vectors $$(\mathbf{v}_1)$$ and $$(\mathbf{v}_2)$$, 
the **cosine similarity** is defined as:

$$
\cos(\mathbf{v}_1, \mathbf{v}_2) 
= \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|\|\mathbf{v}_2\|}
$$

- **Old method**: Performed a pairwise loop over every pair of vectors, 
  incurring $$(O(N^2 \times D))$$ complexity, which can be slow for large sets.

- **New approach**: We **normalize** each row of \(E\), then compute:

$$
S 
= \bigl(\tfrac{E}{\|E\|}\bigr)\;
  \bigl(\tfrac{E}{\|E\|}\bigr)^{T}
$$

to get **all pairwise similarities** in a single matrix multiplication step.

---

### Notes on RoSE Dataset Integration

This package has been particularly useful in deduplicating “reference ACUs” (Atomic Content Units) from the [**ROSE dataset**](https://github.com/Yale-LILY/ROSE). In ROSE, multiple human-written references can be repetitive or near-duplicates, so performing this claim-based clustering/deduplication:

1. **Reduces redundancy** in the reference sets.
2. **Organizes** similar human-written units into coherent clusters.
3. **Streamlines** the data for downstream tasks or evaluations (e.g., summarization research).

In short, if you have a dataset (like RoSE) with numerous overlapping or near-duplicate human reference texts, running the **claims-deduplicator** pipeline can help produce a cleaner, more concise reference set.


---

## Multi-Threshold Deduplication

If you want to compare results at different thresholds without re-computing all embeddings multiple times, use `multi_threshold_deduplicate`:

```python
from claim_deduplicator.multi_threshold_deduplicate import multi_threshold_deduplicate
from claim_deduplicator.strategies import select_longest

multi_threshold_deduplicate(
    input_json_path="data/example_dataset.json",
    output_json_path="data/multi_threshold_output.json",
    thresholds=[0.7, 0.8, 0.9],
    representative_selector=select_longest,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    measure_redundancy_flag=True,
    cluster_analysis_dir="analysis_results"
)
```

This will produce additional fields like `deduped_0.7_longest`, `deduped_0.8_longest`, etc., in each record.

---

## License

This project is licensed under the [MIT License](./LICENSE).  
You’re free to modify and distribute it as per the license terms.

