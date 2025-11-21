import os

HF_TOKEN = os.environ.get("HF_TOKEN")  # Your HuggingFace API token
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Optional if you use repo
HUGGINGFACE_MODEL_NAME = "mradermacher/medical-20-0-16-jinaai_jina-embeddings-v2-small-en-100-gpt-3.5-turbo-0_9062874564-i1-GGUF"

DB_FAISS_PATH = "vectorstore/db/faiss"  # Fixed spelling
DATA_PATH = "data/"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
