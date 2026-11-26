from pathlib import Path
from fastembed import TextEmbedding

TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=str(Path(__file__).resolve().parent / "models"),
)