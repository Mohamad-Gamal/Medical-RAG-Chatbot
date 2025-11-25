# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from langchain_core.messages import HumanMessage  # For prompt formatting
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from ollama import Ollama
from langchain.llms.base import LLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get model from env
model_name = os.getenv("OLLAMA_MODEL")
api_key = os.getenv("OLLAMA_API_KEY")  # Optional if required

# Initialize Ollama client
# ollama_client = Ollama(api_key=api_key) if api_key else Ollama()

logger = get_logger(__name__)

class OllamaLLM(LLM):
    def __init__(self, model_name, api_key=None):
        self.client = Ollama(api_key=api_key) if api_key else Ollama()
        self.model_name = model_name

    @property
    def _llm_type(self):
        return "ollama"

    def _call(self, prompt, stop=None):
        response = self.client.chat(model=self.model_name, prompt=prompt, temperature=0.7, max_tokens=256)
        return response

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN, prompt: str = ""):
    try:
        logger.info("Loading LLM using HuggingFaceEndpoint...")

        if not hf_token:
            raise CustomException("HF_TOKEN is missing in environment variables.")

        # Use HuggingFaceEndpoint with explicit task parameter
        # endpoint = HuggingFaceEndpoint(
        #     repo_id=huggingface_repo_id,
        #     huggingfacehub_api_token=hf_token,
        #     task="text-generation",  # This is REQUIRED
        #     temperature=0.1,
        #     max_new_tokens=512,
        #     top_p=0.9,
        #     do_sample=True,
        #     return_full_text=False
        # )
        # llm = ChatHuggingFace(llm=endpoint)
        llm = OllamaLLM(model=model_name)
        logger.info(f"LLM loaded successfully: {huggingface_repo_id}")
        return llm

    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"Error loading LLM: {e}")