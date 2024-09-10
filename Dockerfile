# Use uma imagem base do Python que corresponda à versão que você está usando localmente
FROM python:3.12.5-slim-bullseye as builder

WORKDIR /app

# Instalar dependências de compilação
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Baixar e compilar SQLite3
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz && \
    tar xzf sqlite-autoconf-3420000.tar.gz && \
    cd sqlite-autoconf-3420000 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3420000 sqlite-autoconf-3420000.tar.gz

# Baixar e compilar Ghostscript
RUN wget https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10020/ghostscript-10.02.0.tar.gz && \
    tar xzf ghostscript-10.02.0.tar.gz && \
    cd ghostscript-10.02.0 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf ghostscript-10.02.0 ghostscript-10.02.0.tar.gz

FROM python:3.12.5-slim-bullseye

WORKDIR /app

# Copiar SQLite3 e Ghostscript compilados do estágio de compilação
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/ghostscript /usr/local/share/ghostscript

# Copiar o ambiente virtual local
COPY ollama ollama

# Copiar os arquivos do projeto
COPY *.py .
COPY requirements.txt .

# Instalar dependências do sistema necessárias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configurar o ambiente
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV USER_AGENT="RAG"
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=$OLLAMA_HOST
ENV PATH="/app/ollama/Scripts:/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
ENV VIRTUAL_ENV="/app/ollama"

# Ativar o ambiente virtual e verificar/atualizar dependências
RUN python -m venv /app/ollama && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verificar a instalação
RUN . /app/ollama/Scripts/activate && \
    python --version && \
    pip list && \
    python -c "import sqlite3; print('SQLite3 version:', sqlite3.sqlite_version)" && \
    python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)" && \
    gs --version

# Expor a porta da aplicação
EXPOSE 7863

# Comando para executar a aplicação
CMD ["python", "main.py"]