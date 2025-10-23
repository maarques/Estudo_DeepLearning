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

app = Flask(__name__)
app.secret_key = "meu_nome"

contexto = carrega("02-python-e-gpt-crie-seu-chatbot-com-ia\dados\EcoMart.txt")

def bot(prompt):
    max_tentativas = 1
    repeticao = 0
    personalidade = personas[selecionar_persona(prompt)]

    while True:
        try:
            prompt_sistema = f"""
            Você é um chatbot de atendimento a clientes de um e-commerce. 
            Você não deve responder perguntas que não sejam dados do e-commerce informado!

            Você deve gerar respostas utilizando o contexto abaixo.
            Você deve adotar a persona abaixo.

            # Contexto
            {contexto}

            #Persona
            {personalidade}
            """

            response = client.chat.completions.create(
                messages=[
                        {
                                "role": "system",
                                "content": prompt_sistema
                        },
                        {
                                "role": "user",
                                "content": prompt
                        }
                ],
                temperature=1,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model = modelo)
            return response
        except Exception as erro:
                repeticao += 1
                if repeticao >= max_tentativas:
                    return "Erro no GPT: %s" % erro
                print('Erro de comunicação com OpenAI:', erro)
                sleep(1)
             

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json["msg"]
    resposta = bot(prompt)
    texto_resposta = resposta.choices[0].message["content"]
    return texto_resposta

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
