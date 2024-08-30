from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
import os
from config import get_logger
from document_validator import validate_document_context, get_rejection_reason
import ocrmypdf
from pdf2image import convert_from_path
import pytesseract
from transformers import pipeline
import PyPDF2

logger = get_logger(__name__)

# Carregar pipeline de classificação de texto
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Lista de possíveis classes para o tipo de processo e situação
tipo_processo_classes = [
    "Habeas Corpus",
    "Recurso de Habeas Corpus",
    "Mandado de Segurança",
    "Recurso em Mandado de Segurança",
    "Apelação Criminal",
    "Apelação Cível",
    "Agravo de Instrumento",
    "Agravo Regimental",
    "Embargos de Declaração",
    "Recurso Especial",
    "Recurso Extraordinário",
    "Ação Civil Pública",
    "Ação Penal",
    "Inquérito Policial",
    "Medida Cautelar",
    "Liminar",
    "Recurso Ordinário",
    "Conflito de Competência",
    "Suspensão de Segurança",
    "Reclamação",
    "Revisão Criminal",
    "Ação Rescisória",
    "Mandado de Injunção",
    "Ação Direta de Inconstitucionalidade",
    "Ação Declaratória de Constitucionalidade",
    "Outro Processo"
]
situacao_classes = ["Julgado e deferido", "Julgado e indeferido", "Condenado", "Negado", "Acatado", "Em trâmite", "Outro"]

def extract_text_from_pdf(pdf_path, method='ocrmypdf'):
    try:
        if method == 'ocrmypdf':
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_output:
                temp_output_path = temp_output.name

            ocrmypdf.ocr(pdf_path, temp_output_path, language='por')

            with open(temp_output_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages)

            os.remove(temp_output_path)

        elif method == 'pdf2image':
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image, lang='por')
        else:
            raise ValueError("Método de extração inválido")

        if not text.strip():
            logger.warning("Nenhum texto extraído do PDF.")
        else:
            logger.info("Texto extraído do PDF com sucesso.")

        return text
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}", exc_info=True)
        return ""

def classify_case_type(text):
    # Primeiro, tenta classificar com base na lista completa
    result = classifier(text, tipo_processo_classes)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    # Se a pontuação for baixa, tenta uma abordagem mais detalhada
    if top_score < 0.5:  # Você pode ajustar este limiar conforme necessário
        # Verifica se é um recurso
        if "recurso" in text.lower():
            for tipo in tipo_processo_classes:
                if tipo.lower().startswith("recurso") and tipo.lower() in text.lower():
                    return tipo
            return "Outro Recurso"
        
        # Verifica outros tipos específicos
        for tipo in tipo_processo_classes:
            if tipo.lower() in text.lower():
                return tipo

    return top_label

def classify_case_status(text):
    result = classifier(text, situacao_classes)
    return result["labels"][0]

def load_and_split_document(source, extraction_method='ocrmypdf'):
    logger.info(f"Carregando dados do PDF usando o método {extraction_method}")
    if isinstance(source, str):
        pdf_path = source
    else:
        pdf_path = source.name
    text = extract_text_from_pdf(pdf_path, method=extraction_method)
    if not text.strip():
        logger.warning("Não foi possível extrair texto do PDF.")
        return None, "Não foi possível extrair texto do PDF."

    # Validar o contexto do documento
    logger.info("Validando o contexto do documento...")
    is_valid, validation_response = validate_document_context(text)
    if not is_valid:
        logger.warning(f"Documento rejeitado: {validation_response}")
        return None, f"O documento não está relacionado ao contexto jurídico. Razão: {validation_response}"

    logger.info("Documento validado com sucesso.")

    # Classificar o tipo de processo e situação
    tipo_processo = classify_case_type(text)
    situacao = classify_case_status(text)
    logger.info(f"Tipo de processo classificado: {tipo_processo}")
    logger.info(f"Situação classificada: {situacao}")

    docs = [Document(page_content=text, metadata={"source": pdf_path, "tipo_processo": tipo_processo, "situacao": situacao})]

    logger.info(f"Carregados {len(docs)} documentos")

    if not docs:
        logger.warning("Nenhum documento foi carregado.")
        return None, "Nenhum documento foi carregado."

    logger.info("Dividindo documentos em chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    logger.info(f"Criados {len(splits)} splits")

    return splits, None  # None para o erro indica sucesso