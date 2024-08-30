from datetime import datetime
import os
from config import get_logger
from document_processor import load_and_split_document
from embedding_manager import create_embeddings
from chroma_manager import create_collection
from rag_engine import rag_chain
from faq import FAQ
import uuid

logger = get_logger(__name__)

def load_context(source, extraction_method='ocrmypdf'):
    if not source:
        yield {"status": "Por favor, faça upload de um PDF.", "success": False, "collection": None}
        return

    try:
        yield {"status": "Iniciando carregamento e validação do documento...", "success": False, "collection": None}
        splits, error_message = load_and_split_document(source, extraction_method=extraction_method)
        if error_message:
            yield {"status": error_message, "success": False, "collection": None}
            return

        if not splits:
            yield {"status": "O documento está fora do contexto esperado!", "success": False, "collection": None}
            return

        yield {"status": "Documento validado. Criando embeddings...", "success": False, "collection": None}
        embeddings = create_embeddings()
        collection_name = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        collection = create_collection(collection_name, embeddings)

        yield {"status": "Embeddings criados. Adicionando documentos à coleção...", "success": False, "collection": None}
        ids = [str(uuid.uuid4()) for _ in splits]
        collection.add(
            documents=[split.page_content for split in splits],
            metadatas=[split.metadata for split in splits],
            ids=ids
        )

        final_status = f"Contexto criado com sucesso. Pronto para perguntas!"
        yield {"status": final_status, "success": True, "collection": collection}
    except Exception as e:
        logger.error(f"Erro ao processar documento: {str(e)}", exc_info=True)
        yield {"status": f"Ocorreu um erro ao processar o documento: {str(e)}", "success": False, "collection": None}

def answer_question(question, collection, model, llm_interface):
    if not collection:
        return "Por favor, processe um documento jurídico válido antes de fazer perguntas."

    try:
        answer = rag_chain(question, model, collection, llm_interface)
        return answer
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}", exc_info=True)
        return f"Ocorreu um erro ao processar a pergunta: {str(e)}"

def process_faq(collection, model, llm_interface):
    if not collection:
        return "Por favor, processe um documento jurídico válido antes de executar o FAQ."

    try:
        faq = FAQ()
        faq_results = faq.get_faq_answers(collection, model, llm_interface)
        return faq_results['questions_and_answers']
    except Exception as e:
        logger.error(f"Erro ao processar FAQ: {str(e)}", exc_info=True)
        return f"Ocorreu um erro ao processar o FAQ: {str(e)}"