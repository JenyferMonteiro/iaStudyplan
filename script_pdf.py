import os
import PyPDF2
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Carrega variáveis do .env
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

# Carrega todos os PDFs da pasta
documentos = carregar_varios_pdfs("pdf")

# Usa um modelo válido da Groq (Llama 3.2)
agente = Agent(
    model=Groq(id="openai/gpt-oss-20b"),  # modelo atualizado e suportado
    instructions="Responda apenas com base nos PDFs fornecidos."
    "Escreva a resposta em texto corrido, direto e objetivo, "
    "sem usar tabelas. Use no máximo um ou dois parágrafos curtos, "
    "e cite o nome do documento de onde tirou a informação."
)

while True:
    pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        break

    contexto = ""
    for nome, texto in documentos.items():
        chunks = dividir_texto(texto)
        contexto += f"\n--- {nome} ---\n" + "\n".join(chunks[:3])

    resposta = agente.run(f"Com base nos seguintes textos: {contexto}\n\nPergunta: {pergunta}")
    print("\nResposta:", resposta.content)