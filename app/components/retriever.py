from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompt_values import PromptTamplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.config.config import HF_TOKEN, HUGGINGFACE_MODEL_NAME, DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException 
import os

logger = get_logger(__name__)   

custom_prompt_template = """
    You are a highly intelligent question answering bot. 
    Answer the following medical question in 2-3 lines maximum using only the information provided in the context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Answer:"""


def set_custom_prompt(qa_chain: RetrievalQA) -> RetrievalQA:
    try:
        logger.info("Setting custom prompt template for the QA chain...")

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template= custom_prompt_template
            )
        logger.info("Custom prompt template set successfully.")
        return prompt
    except Exception as e:
        logger.error(f"Error setting custom prompt template: {e}")
        raise CustomException(f"Error setting custom prompt template: {e}")

def get_retriever_qa(llm: HuggingFaceHub) -> RetrievalQA:
    try:
        logger.info("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_MODEL_NAME,
            huggingfacehub_api_token=HF_TOKEN
        )
        logger.info("Embeddings loaded successfully.")

        logger.info("Loading LLM...")
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
        logger.info("LLM loaded successfully.")
        if llm is None:
            raise CustomException("LLM could not be loaded.")

        logger.info("Loading FAISS vector store...")
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings)
        logger.info("FAISS vector store loaded successfully.")
        if vector_store is None:
            raise CustomException("Vector store could not be loaded.")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logger.info("Creating RetrievalQA chain...")

        # Retriver chain 
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                retriever=retriever, return_source_documents=True,
                                                chain_type_kwargs={"prompt": set_custom_prompt})
        logger.info("RetrievalQA chain created successfully.")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating RetrievalQA chain: {e}")
        raise CustomException(f"Error creating RetrievalQA chain: {e}")