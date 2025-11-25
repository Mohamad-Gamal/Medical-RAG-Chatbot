import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)

def load_csv_files():
    """
    Load all CSV files from DATA_PATH and convert each row into a Document.
    """
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data Path does not exist: {DATA_PATH}")

        logger.info(f"Loading CSV files from {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="*.csv", loader_cls=CSVLoader)
        documents = loader.load()

        if not documents:
            logger.warning("No CSV documents found")
        else:
            logger.info(f"Successfully loaded {len(documents)} CSV documents")

        return documents

    except Exception as e:
        error_message = CustomException("Failed to load CSVs!", e)
        logger.error(str(error_message))
        return []

def create_text_chunks(documents):
    """
    Split documents into text chunks for vector embeddings.
    """
    try:
        if not documents:
            raise CustomException("No documents were found!")

        logger.info(f"Splitting {len(documents)} documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks

    except Exception as e:
        error_message = CustomException("Failed to generate text chunks!", e)
        logger.error(str(error_message))
        return []

def load_and_prepare_csv_chunks():
    """
    Full workflow: load CSV files and split into chunks ready for FAISS vectorization.
    """
    documents = load_csv_files()
    chunks = create_text_chunks(documents)
    return chunks