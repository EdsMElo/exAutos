from typing import Dict, List
from config import get_logger
from rag_engine import rag_chain

logger = get_logger(__name__)

class FAQ:
    def __init__(self):
        self.questions = [
            "Qual é o tipo específico de recurso ou processo judicial mencionado nos autos?",
            "Qual é a situação atual do processo?",
            "Quais são os próximos passos processuais ou as consequências imediatas desta decisão para as partes envolvidas?"
        ]

    def get_faq_answers(self, collection, model, llm_interface) -> Dict[str, str]:
        logger.info("Iniciando processo de FAQ sequencial")
        
        answers = []
        context = ""
        tipo_processo = None
        situacao = None

        # Obter metadados do primeiro documento (assumindo que todos os documentos têm os mesmos metadados)
        first_doc = collection.get(limit=1)
        if first_doc and first_doc['metadatas']:
            tipo_processo = first_doc['metadatas'][0].get('tipo_processo')
            situacao = first_doc['metadatas'][0].get('situacao')

        for i, question in enumerate(self.questions):
            logger.info(f"Processando pergunta {i+1}")

            if i == 0 and tipo_processo:
                answers.append(f"O tipo de processo/recurso identificado é: {tipo_processo}")
                context += f"\nPergunta: {question}\nResposta: {answers[-1]}\n"
                continue

            if i == 1 and situacao:
                answers.append(f"A situação atual do processo é: {situacao}")
                context += f"\nPergunta: {question}\nResposta: {answers[-1]}\n"
                continue

            prompt = f"""
            Analise cuidadosamente o documento jurídico fornecido e responda à seguinte pergunta:

            {question}

            Contexto adicional:
            Tipo de processo: {tipo_processo}
            Situação: {situacao}
            {context}

            Instruções:
            - Forneça uma resposta clara e concisa, baseada nas informações do documento.
            - Se a informação exata não estiver disponível, forneça a informação mais próxima ou relevante que puder encontrar.
            - Se não houver absolutamente nenhuma informação relevante, responda "Informação não disponível no documento."

            Resposta:
            """
            
            logger.info(f"Enviando prompt para o modelo: {prompt[:200]}...")
            answer = rag_chain(prompt, model, collection, llm_interface)
            logger.info(f"Resposta obtida: {answer[:200]}...")
            
            answer = self.clean_and_validate_answer(answer)
            answers.append(answer)
            
            context += f"\nPergunta: {question}\nResposta: {answer}\n"

        formatted_result = self.format_results(self.questions, answers)
        logger.info("Processo de FAQ sequencial concluído")
        return formatted_result

    def clean_and_validate_answer(self, answer: str) -> str:
        answer = answer.strip()
        if answer.lower().startswith("resposta:"):
            answer = answer[9:].strip()
        if not answer or answer.lower() == "resposta:" or "pergunta:" in answer.lower():
            return "Informação não disponível no documento."
        return answer

    def format_results(self, questions: List[str], answers: List[str]) -> Dict[str, str]:
        formatted_qa = "\n\n".join([f"P: {q}\nR: {a}" for q, a in zip(questions, answers)])
        return {"questions_and_answers": formatted_qa}