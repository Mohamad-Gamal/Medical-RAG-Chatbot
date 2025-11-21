from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains import create_retrieval_chain

from app.config.config import HF_TOKEN, HUGGINGFACE_MODEL_NAME, HUGGINGFACE_REPO_ID, DB_FAISS_PATH
from app.components.llm import load_llm
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


custom_prompt_template = """
You are a highly intelligent medical question-answering assistant.
Answer the medical question in 5-7 lines using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{input}

Answer:
"""


def get_retriever_qa():
    try:
        # Load embeddings
        logger.info("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)

        # Load FAISS vector store
        logger.info("Loading FAISS vector store...")
        vector_store = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Load LLM
        logger.info("Loading LLM...")
        llm = load_llm(
            huggingface_repo_id=HUGGINGFACE_REPO_ID,
            hf_token=HF_TOKEN
        )

        if llm is None:
            raise CustomException("LLM failed to load.")

        # Build Prompt
        prompt = PromptTemplate.from_template(custom_prompt_template)

        # Build document-chain using classic LangChain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Build the full RAG retrieval chain
        logger.info("Creating Retrieval chain (RAG)...")
        rag_chain = create_retrieval_chain(retriever, document_chain)

        logger.info("RAG chain created successfully.")
        return rag_chain

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        raise CustomException(f"Error creating RAG chain: {e}")
