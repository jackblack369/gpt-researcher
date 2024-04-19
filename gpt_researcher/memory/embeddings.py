from langchain_community.vectorstores import FAISS
import os


class Memory:
    def __init__(self, embedding_provider, **kwargs):
        self.api_key = self.get_api_key()
        self.base_url = self.get_base_url()

        _embeddings = None
        match embedding_provider:
            case "ollama":
                from langchain.embeddings import OllamaEmbeddings
                _embeddings = OllamaEmbeddings(model="llama2")
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(api_key=self.api_key,
                                               base_url=self.base_url)
            case "azureopenai":
                from langchain_openai import AzureOpenAIEmbeddings
                _embeddings = AzureOpenAIEmbeddings(deployment=os.environ["AZURE_EMBEDDING_MODEL"], chunk_size=16)
            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings
                _embeddings = HuggingFaceEmbeddings()

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_api_key(self):
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return api_key

    def get_base_url(self):
        try:
            base_url = os.environ["OPENAI_BASE_URL"]
        except:
            raise Exception(
                "OpenAI base url not found. Please set the OPENAI_BASE_URL environment variable.")
        return base_url

    def get_embeddings(self):
        return self._embeddings
