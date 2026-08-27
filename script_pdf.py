import os
import re
import numpy as np
import faiss
import PyPDF2
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def carregar_varios_pdfs(pasta="pdf"):
    documentos = {}
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            texto = ""
            with open(caminho, "rb") as f:
                leitor = PyPDF2.PdfReader(f)
                for pagina in leitor.pages:
                    texto += pagina.extract_text()
            documentos[arquivo] = texto
    return documentos

def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())

def construir_indice(chunks):
    """
    Constrói um índice FAISS a partir dos chunks de texto usando TF-IDF.
    Isso permite buscar, para cada pergunta, os pedaços de PDF realmente
    relevantes, em vez de sempre pegar os primeiros pedaços do documento.
    """
    vocabulario = {}
    docs_tokens = [tokenizar(c) for c in chunks]
    for tokens in docs_tokens:
        for termo in set(tokens):
            if termo not in vocabulario:
                vocabulario[termo] = len(vocabulario)

    n_docs = len(chunks)
    n_termos = len(vocabulario)

    tf = np.zeros((n_docs, n_termos), dtype="float32")
    for i, tokens in enumerate(docs_tokens):
        for termo in tokens:
            tf[i, vocabulario[termo]] += 1

    df = (tf > 0).sum(axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    tfidf = tf * idf

    normas = np.linalg.norm(tfidf, axis=1, keepdims=True)
    normas[normas == 0] = 1
    tfidf = tfidf / normas

    indice = faiss.IndexFlatIP(n_termos)
    indice.add(tfidf)
    return indice, vocabulario

def vetorizar_pergunta(pergunta, vocabulario):
    n_termos = len(vocabulario)
    vetor = np.zeros((1, n_termos), dtype="float32")
    for termo in tokenizar(pergunta):
        if termo in vocabulario:
            vetor[0, vocabulario[termo]] += 1
    norma = np.linalg.norm(vetor)
    if norma > 0:
        vetor = vetor / norma
    return vetor

def buscar_chunks_relevantes(pergunta, indice, vocabulario, chunks_info, top_k=5):
    """Retorna os top_k chunks (nome_do_doc, texto) mais relevantes para a pergunta."""
    vetor = vetorizar_pergunta(pergunta, vocabulario)
    _, posicoes = indice.search(vetor, min(top_k, len(chunks_info)))
    return [chunks_info[i] for i in posicoes[0] if i != -1]

documentos = carregar_varios_pdfs("pdf")

chunks_info = []
for nome, texto in documentos.items():
    for chunk in dividir_texto(texto):
        if chunk.strip():
            chunks_info.append((nome, chunk))

textos_chunks = [texto for _, texto in chunks_info]
indice, vocabulario = construir_indice(textos_chunks) if textos_chunks else (None, None)

agente = Agent(
    model=Groq(
        id="openai/gpt-oss-20b",  
        temperature=0.2,          
        retries=2,                
        delay_between_retries=2,
        exponential_backoff=True,
    ),
    instructions="Responda apenas com base nos PDFs fornecidos."
    "Escreva a resposta em texto corrido, direto e objetivo, "
    "sem usar tabelas. Use no máximo um ou dois parágrafos curtos, "
    "e cite o nome do documento de onde tirou a informação."
)

while True:
    pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        break

    if indice is not None:
        relevantes = buscar_chunks_relevantes(pergunta, indice, vocabulario, chunks_info, top_k=5)
    else:
        relevantes = []

    contexto = ""
    for nome, texto in relevantes:
        contexto += f"\n--- {nome} ---\n{texto}"

    if not contexto:
        print("\nResposta: Não encontrei nenhum PDF carregado para responder essa pergunta.")
        continue

    try:
        resposta = agente.run(f"Com base nos seguintes textos: {contexto}\n\nPergunta: {pergunta}")
        print("\nResposta:", resposta.content)
    except Exception as erro:
        print(f"\nOcorreu um erro ao consultar o modelo, tente novamente: {erro}")