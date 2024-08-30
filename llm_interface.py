from config import OLLAMA_HOST, get_logger
from ollama import Client

logger = get_logger(__name__)

def ollama_llm(question, context, model="phi3.5:3.8b"):
    logger.info(f"Chamando Ollama LLM com a pergunta: {question} usando o modelo: {model}")
    
    try:
        ollama = Client(host=OLLAMA_HOST)
        
        prompt = f"""Instruções: Responda à pergunta com base no contexto fornecido. Responda sempre em português do Brasil.

Contexto: {context}

Pergunta: {question}

Resposta em português do Brasil:"""
        
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': 'Você é um assistente que responde exclusivamente em português do Brasil.'},
            {'role': 'user', 'content': prompt}
        ])
        logger.info("Resposta recebida do Ollama LLM")
        
        full_response = response['message']['content']
        answer_prefix = "Resposta em português do Brasil:"
        if answer_prefix in full_response:
            clean_response = full_response.split(answer_prefix)[-1].strip()
        else:
            clean_response = full_response.strip()
        
        return clean_response
    except Exception as e:
        logger.error(f"Erro ao chamar Ollama LLM: {str(e)}", exc_info=True)
        return f"Ocorreu um erro ao usar o Ollama: {str(e)}"

def get_available_models():
    return ["phi3.5:3.8b"]