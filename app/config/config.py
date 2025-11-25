import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# HuggingFace Tokens
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# LLM model (for answering questions)
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen-2:1.5b")

# Embedding model (for FAISS vector store)
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Paths
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db/faiss")
DATA_PATH = os.getenv("DATA_PATH", "data/")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
