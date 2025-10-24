from flask import Flask, render_template, request, Response
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from time import sleep
from helpers import *
from selecionar_persona import *

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

politicas_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia\dados\políticas_ecomart.txt")
dados_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia\dados\dados_ecomart.txt")
produtos_ecomart = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia\dados\produtos_ecomart.txt")

def selecionar_documento(resposta_huggingface):
    if "políticas" in resposta_huggingface:
        return dados_ecomart + "\n" + politicas_ecomart
    elif "produtos" in resposta_huggingface:
        return dados_ecomart + "\n" + produtos_ecomart
    else:
        return dados_ecomart

def selecionar_contexto(msg_usuario):
    prompt_sistema = f"""
    A empresa EcoMart possui três documentos principais que detalham diferentes aspectos do negócio:

    #Documento 1 "\n {dados_ecomart} "\n"
    #Documento 2 "\n" {politicas_ecomart} "\n"
    #Documento 3 "\n" {produtos_ecomart} "\n"

    Avalie o prompt do usuário e retorne o documento mais indicado para ser usado no contexto da resposta. 
    Retorne dados se for o Documento 1, políticas se for o Documento 2 e produtos se for o Documento 3. 

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
        temperature=1
    )
    contexto = resposta.choices[0].message["content"].lower()
    return contexto