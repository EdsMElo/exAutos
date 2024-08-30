import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import get_logger
from embedding_manager import create_embeddings

logger = get_logger(__name__)

def semantic_search(query, documents, top_k=10):
    vectorizer = TfidfVectorizer().fit([query] + documents)
    doc_vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    return similarities.argsort()[-top_k:][::-1]

def dynamic_chunk_selection(question, chunks, initial_k=5, max_k=20, similarity_threshold=0.3):
    selected_chunks = []
    k = initial_k

    while k <= max_k:
        top_chunk_indices = semantic_search(question, chunks, top_k=k)
        selected_chunks = [chunks[i] for i in top_chunk_indices]
        
        if np.mean([cosine_similarity(TfidfVectorizer().fit_transform([question, chunk])) for chunk in selected_chunks]) >= similarity_threshold:
            break
        
        k += 5

    return selected_chunks

def rag_chain(question, model, collection, llm_interface):
    logger.info(f"Processando pergunta: {question} com o modelo: {model}")
    
    embedding_function = create_embeddings()
    query_embedding = embedding_function(question)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    
    relevant_chunks = dynamic_chunk_selection(question, results['documents'][0])
    
    logger.info(f"Selecionados {len(relevant_chunks)} chunks relevantes")
    
    # Limitar o número de chunks para controlar o tamanho do contexto
    max_chunks = 3  # Ajuste este valor conforme necessário
    context = "\n\n".join(relevant_chunks[:max_chunks])
    
    prompt = f"""Instruções: Responda à pergunta com base APENAS nas informações fornecidas no contexto abaixo. 
    Se a informação necessária não estiver no contexto, diga que não pode responder com base nas informações disponíveis.

    Contexto:
    {context}

    Pergunta: {question}

    Instruções adicionais:
    1. Use APENAS as informações do contexto acima para responder.
    2. Não use conhecimentos externos ou suposições além do que está no contexto.
    3. Se o contexto não fornecer informações suficientes, diga isso claramente.
    4. Estruture sua resposta de forma clara e concisa.
    5. Se apropriado, cite partes específicas do contexto para apoiar sua resposta.

    Resposta:"""

    response = llm_interface(question, prompt, model)
    
    return response