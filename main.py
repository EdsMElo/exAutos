from config import get_logger
from prompt_manager import load_context, answer_question, process_faq
from llm_interface import ollama_llm, MODEL_PROMPT
from frontend import create_interface, launch_interface

logger = get_logger(__name__)

def main():
    logger.info("Iniciando a aplicação")

    collection = None
    model = MODEL_PROMPT
    extraction_method = "ocrmypdf"  # Padrão, pode ser alterado para "pdf2image"

    def load_context_wrapper(*args):
        nonlocal collection
        for result in load_context(*args, extraction_method=extraction_method):
            if result["success"]:
                collection = result["collection"]
            yield result

    def answer_question_wrapper(question):
        return answer_question(question, collection, model, ollama_llm)

    def process_faq_wrapper():
        return process_faq(collection, model, ollama_llm)

    def set_extraction_method(method):
        nonlocal extraction_method
        extraction_method = method
        logger.info(f"Método de extração definido para: {extraction_method}")

    iface = create_interface(
        load_context_wrapper,
        answer_question_wrapper,
        process_faq_wrapper,
        set_extraction_method
    )

    launch_interface(iface)

if __name__ == "__main__":
    main()