from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from helpers import *
from selecionar_persona import *

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

try:
    politicas_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia/dados/políticas_ecomart.txt")
    dados_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia/dados/dados_ecomart.txt")
    produtos_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia/dados/produtos_ecomart.txt")
    
    Documentos_Ecomart = {
        "políticas": dados_ecomart + "\n" + politicas_ecomart,
        "produtos": dados_ecomart + "\n" + produtos_ecomart,
        "dados": dados_ecomart
    }
    
    print("Arquivos de contexto carregados com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar os arquivos de dados: {e}")
    Documentos_Ecomart = {"políticas": "", "produtos": "", "dados": ""}

def selecionar_documento(nome_contexto):

    nome_normalizado = nome_contexto.lower().strip().replace(".", "")
    
    if nome_normalizado not in Documentos_Ecomart:
        nome_normalizado = "dados"
        
    return Documentos_Ecomart[nome_normalizado]


def selecionar_contexto(msg_usuario):

    prompt_sistema = f"""
    A empresa EcoMart possui três documentos principais que detalham diferentes aspectos do negócio:

    #Documento 1 (retorne "dados")
    {dados_ecomart}

    #Documento 2 (retorne "políticas")
    {politicas_ecomart}

    #Documento 3 (retorne "produtos")
    {produtos_ecomart}

    Avalie o prompt do usuário e retorne APENAS a palavra-chave do documento mais indicado para ser usado no contexto da resposta.
    Retorne "dados" se for o Documento 1, "políticas" se for o Documento 2 e "produtos" se for o Documento 3.
    """
    
    resposta = client.chat.completions.create(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": prompt_sistema
            },
            {
                "role": "user",
                "content": msg_usuario
            }
        ],
        temperature=0
    )
    contexto = resposta.choices[0].message["content"].lower().strip().replace(".", "")
    return contexto
