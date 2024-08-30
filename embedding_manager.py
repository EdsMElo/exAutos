from langchain_community.embeddings import OllamaEmbeddings
from config import OLLAMA_HOST, get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"

class ChromaCompatibleEmbedding:
    def __init__(self):
        self.model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)

    def __call__(self, input):
        if isinstance(input, str):
            return self.model.embed_query(input)
        elif isinstance(input, list):
            return self.model.embed_documents(input)
        else:
            raise ValueError("Input deve ser uma string ou uma lista de strings")

def create_embeddings():
    logger.info(f"Criando embeddings Ollama compatíveis com Chroma usando o modelo {EMBEDDING_MODEL}")
    return ChromaCompatibleEmbedding()