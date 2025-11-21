from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

from app.config.config import HF_TOKEN, HUGGINGFACE_MODEL_NAME, HUGGINGFACE_REPO_ID, DB_FAISS_PATH
from app.components.llm import load_llm
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

custom_prompt_template = """
You are a highly intelligent medical question-answering assistant.
Answer the medical question clearly and concisely using ONLY the provided context.
If the answer is not in the context, politely state that you cannot answer based on the available information.

Context:
{context}

Question:
{question}

Please provide a helpful and accurate medical response:
"""

def get_retriever_qa():
    try:
        logger.info("Initializing medical QA system...")

        # Load embeddings
        logger.info("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_MODEL_NAME,
            model_kwargs={'device': 'cpu'}  # Change to 'cuda' if GPU available
        )

        # Load FAISS vector store
        logger.info("Loading FAISS vector store...")
        vector_store = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Create retriever
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

        # Test the LLM
        try:
            test_response = llm.invoke("Say 'Medical AI ready' in one word.")
            logger.info(f"LLM test response: {test_response}")
        except Exception as e:
            logger.warning(f"LLM test failed, but continuing: {e}")

        # Build prompt template (for retrieval QA)
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )

        # Create Conversational RetrievalQA chain
        logger.info("Creating ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,          # conversational LLM
            retriever=retriever,
            return_source_documents=True
        )

        logger.info("Medical QA system initialized successfully.")
        return qa_chain

    except Exception as e:
        logger.error(f"Error creating medical QA system: {e}")
        raise CustomException(f"Error creating medical QA system: {e}")
