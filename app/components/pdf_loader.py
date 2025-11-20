import os
from langchain_community.document_loaders import DirectoryLoader, PyPDF_Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import Custom_Exception
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise Custom_Exception("Data Path does not exist")

        logger.info(f"Loading files from{DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH,glob="*.pdf",loader_cls=PyPDF_Loader)

        documents = loader.load()

        if not documents:
            logger.warning("")