from flask import Flask, render_template, request, Response
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from helpers import *
from selecionar_persona import *
from selecionar_documento import *
from assistente_ecomart import *

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"
contexto_inicial = selecionar_documento("02-python-e-gpt-crie-seu-chatbot-com-ia\dados\EcoMart.txt")
assistente_ia = criar_assistente("assistenteEcomart", personas["neutro"], contexto_inicial, modelo)

app = Flask(__name__)
app.secret_key = "meu_nome"

def bot(prompt):
    sentimento = selecionar_persona(prompt)
    if sentimento not in personas:
        sentimento = 'neutro'
    personalidade = personas[sentimento]
    contexto = selecionar_contexto(prompt)
    documento_selecionado = selecionar_documento(contexto)

    messages = [
        {"role": "system", "content": assistente_ia.instructions},
        {"role": "user", "content": prompt}
    ]

    try:
        response = assistente_ia.client.chat.completions.create(
            model=assistente_ia.model,
            messages=messages,
            temperature=1,
            max_tokens=300
        )
        return response.choices[0].message["content"]

    except Exception as erro:
        print("Erro no modelo:", erro)
        return f"Erro no GPT: {erro}"

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json["msg"]
    resposta = bot(prompt)
    return resposta

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
