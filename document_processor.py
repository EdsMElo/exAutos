from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
import os
from config import get_logger
from document_validator import validate_document_context, get_rejection_reason
import ocrmypdf
from pdf2image import convert_from_path
import pytesseract
import PyPDF2
from llm_interface import ollama_llm, MODEL_PROMPT

logger = get_logger(__name__)

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
    prompt = f"""Classifique o tipo de processo jurídico com base no seguinte texto. Escolha a opção mais apropriada entre as seguintes categorias:

{', '.join(tipo_processo_classes)}

Se nenhuma categoria se aplicar exatamente, escolha 'Outro Processo'.

Texto para classificação:
{text[:500]}  // Limitando a 500 caracteres para evitar tokens excessivos

Responda apenas com o nome da categoria escolhida, sem explicações adicionais."""

    response = ollama_llm(prompt, "", MODEL_PROMPT)
    return response.strip()

def classify_case_status(text):
    prompt = f"""Classifique a situação atual do processo jurídico com base no seguinte texto. Escolha a opção mais apropriada entre as seguintes categorias:

{', '.join(situacao_classes)}

Se nenhuma categoria se aplicar exatamente, escolha 'Outro'.

Texto para classificação:
{text[:500]}  // Limitando a 500 caracteres para evitar tokens excessivos

Responda apenas com o nome da categoria escolhida, sem explicações adicionais."""

    response = ollama_llm(prompt, "", MODEL_PROMPT)
    return response.strip()

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