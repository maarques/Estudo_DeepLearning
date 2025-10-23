from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from transformers import AutoTokenizer
import tiktoken
import os

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

codificator = tiktoken.get_encoding("cl100k_base")

def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, "r", encoding="utf-8") as arquivo:
            return arquivo.read()
    except IOError as e:
        print(f"Erro: {e}")

prompt_system = """
Identifique o perfil de compra para cada cliente a seguir.

O formato de saída deve ser:

cliente - descreva o perfil do cliente em 3 palavras
"""

prompt_user = carrega("dados/lista_de_compras_100_clientes.csv")

lista_de_tokens = codificator.encode(prompt_system + prompt_user)
numero_de_tokens = len(lista_de_tokens)
print(f"Número de tokens na entrada: {numero_de_tokens}")

tamanho_esperado_saida = 2048

if numero_de_tokens >= 4096 - tamanho_esperado_saida:
    modelo = "zai-org/GLM-4.6"

print(f"Modelo escolhido: {modelo}")

lista_mensagens = [
    {"role": "system", "content": prompt_system},
    {"role": "user", "content": prompt_user}
]

output = client.chat.completions.create(
    messages=lista_mensagens,
    model=modelo
)

print(output.choices[0].message["content"])
