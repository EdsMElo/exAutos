import chromadb
from config import VECTOR_STORE_DIR, get_logger, CHROMA_SETTINGS

logger = get_logger(__name__)

chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR, settings=CHROMA_SETTINGS)

def create_collection(collection_name, embedding_function):
    logger.info(f"Criando ou obtendo coleção Chroma: {collection_name}")
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )