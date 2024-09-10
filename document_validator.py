from config import get_logger
from llm_interface import ollama_llm

logger = get_logger(__name__)

def validate_document_context(text_sample):
    logger.info("Iniciando validação do contexto do documento...")
    # Reduzindo para os primeiros 2000 caracteres para análise mais abrangente
    sample = text_sample[:2000]
    prompt = f"""Analise cuidadosamente o seguinte texto e determine se ele pertence ao contexto de processos legais, judiciais ou jurídicos.
    Responda apenas 'SIM' ou 'NÃO', seguido de uma breve justificativa.

    Critérios para considerar o texto como jurídico:
    1. Contém terminologia jurídica específica (ex: "habeas corpus", "mandado", "sentença", "réu", "juiz", "recurso", "denúncia").
    2. Faz referências a leis, códigos ou procedimentos judiciais.
    3. Descreve ou discute processos legais, decisões judiciais ou questões de direito.
    4. Menciona instituições jurídicas como tribunais, varas, ou órgãos do judiciário.
    5. Cita artigos de leis ou códigos penais.
    6. Menciona crimes ou infrações legais.

    O texto NÃO é jurídico se:
    1. Apenas menciona termos como "lei" ou "justiça" em contextos não legais.
    2. É um texto de ficção ou entretenimento que apenas menciona elementos jurídicos superficialmente.
    3. É um texto sobre regras de jogos, mesmo que use palavras como "penalidade" ou "julgamento".

    Texto para análise:
    {sample}

    O texto está relacionado a processos legais, judiciais ou jurídicos? Responda 'SIM' ou 'NÃO' e justifique brevemente."""

    response = ollama_llm(prompt, "")
    
    # Extrair a resposta SIM/NÃO
    is_valid = response.strip().upper().startswith('SIM')
    
    logger.info(f"Validação do documento: {'Aprovado' if is_valid else 'Rejeitado'}")
    logger.info(f"Resposta do modelo: {response}")
    
    return is_valid, response

def get_rejection_reason(text_sample):
    logger.info("Obtendo razão para rejeição do documento...")
    sample = text_sample[:2000]  # Usando os primeiros 2000 caracteres para análise mais abrangente
    prompt = f"""Analise o seguinte texto e explique brevemente por que ele não está relacionado a processos legais, judiciais ou jurídicos.
    
    Texto para análise:
    {sample}
    
    Por que este texto não está relacionado ao contexto jurídico? Seja conciso em sua explicação."""

    response = ollama_llm(prompt, "")
    logger.info(f"Razão para rejeição: {response[:200]}...")  # Log dos primeiros 200 caracteres da razão
    return response