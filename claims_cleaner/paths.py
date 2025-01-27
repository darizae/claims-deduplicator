from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class EmbeddingCachePaths:
    """
    Defines where we store embedding cache files for each model.
    If the user picks an unknown model, we create a generic filename.
    """
    cache_dir: Path = BASE_DIR / "cache"

    def get_cache_file_for_model(self, model_name: str) -> Path:
        """
        Return a Path object for the given model name.
        If it's not recognized, we do a simple fallback with
        model_name's slashes replaced by underscores.
        """
        # Create the directory if needed
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # A simple map or dictionary for known models
        if model_name == "sentence-transformers/all-MiniLM-L6-v2":
            return self.cache_dir / "embedding_cache_all_MiniLM.pkl"
        elif model_name == "sentence-transformers/all-mpnet-base-v2":
            return self.cache_dir / "embedding_cache_mpnet.pkl"
        else:
            # fallback
            return self.cache_dir / f"embedding_cache_{model_name.replace('/', '_')}.pkl"
