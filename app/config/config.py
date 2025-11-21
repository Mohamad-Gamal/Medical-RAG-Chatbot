import os

HF_TOKEN = os.environ.get("HF_TOKEN")  # Your HuggingFace API token
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Optional if you use repo
HUGGINGFACE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Example model name

DB_FAISS_PATH = "vectorstore/db/faiss"  # Fixed spelling
DATA_PATH = "data/"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
