# RAG Colosseum

FastAPI retrieval service for the 50-passage RAG Colosseum knowledge base.

## Render deployment

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check path (optional): `/health`

The service exposes:
- `GET /` and `GET /health`
- `POST /query`

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, loaded through FastEmbed/ONNX Runtime to reduce memory use compared with a PyTorch-based SentenceTransformers installation.

Set `RAG_THRESHOLD` as an environment variable if retrieval abstention needs tuning. Default: `0.38`.

The first deployment downloads the embedding model, so the initial build/start may take longer than subsequent starts.
