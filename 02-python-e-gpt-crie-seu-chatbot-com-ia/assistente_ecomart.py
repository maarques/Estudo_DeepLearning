from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import uuid
import json
from helpers import *
from selecionar_persona import *
from selecionar_documento import *

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

contexto = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia/dados/EcoMart.txt")

class Assistente:
    def __init__(self, name, instructions, model):
        self.id = str(uuid.uuid4())
        self.name = name
        self.instructions = instructions
        self.model = model
        self.client = InferenceClient(model=self.model, api_key=os.getenv("HUGGINGFACE_TOKEN_KEY"))
        self.historico = []

    def enviar_mensagem(self, mensagem_usuario):
        self.historico.append({"role": "user", "content": mensagem_usuario})

        resposta = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                *self.historico
            ],
            temperature=0.7,
            max_tokens=512
        )

        conteudo_resposta = resposta.choices[0].message["content"]
        self.historico.append({"role": "assistant", "content": conteudo_resposta})
        return conteudo_resposta

    def mostrar_historico(self):
        for msg in self.historico:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")

def criar_assistente(nome, persona, contexto, modelo):
    assistente_ia = Assistente(
        name="assistenteEcomart",
        instructions=f"""
            Você é um chatbot de atendimento a clientes de um e-commerce. 
            Você não deve responder perguntas que não sejam dados do ecommerce informado!
            Além disso, adote a persona abaixo para responder ao cliente.
            
            ## Contexto
            {contexto}
            
            ## Persona
            {persona}
        """,
        model=modelo
    )
    return assistente_ia

