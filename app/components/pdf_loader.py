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
            logger.warning("There are no pdf")
        
        else:
            logg.info(f"Successfully load {len(documents)} documents")

    except Exception as e:
        error_message = Custom_Exception("Failed to load pdf!", e)
        logger.error(str(error_message))
        return []


def create_text_chunk(documents):
    try:
        if not documents:
            raise Custom_Exception("No documents were found!")


        logger.info(f"Splitting {len(documents)} into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_chunks = text_splitter.split_documents(documents)

        
        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks

    except Exception as e:
        error_message = Custom_Exception("Failed to Generate text chunks!", e)
        logger.error(str(error_message))
        return []