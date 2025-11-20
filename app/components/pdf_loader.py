import os
from langchain_community.document_loaders import DirectoryLoader, PyPDF_Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import Custom_exception
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)
