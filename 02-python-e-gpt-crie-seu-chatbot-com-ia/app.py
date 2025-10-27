from flask import Flask, render_template, request, Response, session
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from helpers import *
from selecionar_persona import *
from selecionar_documento import *

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

app = Flask(__name__)
app.secret_key = "meu_nome"

@app.route("/chat", methods=["POST"])
def chatbot():
    try:
        # Pega o histórico da sessão ou cria um novo se for a primeira mensagem
        historico = session.get("historico", [])
        
        prompt_usuario = request.json["msg"]
        
        # Adiciona a nova mensagem do usuário ao "Thread"
        historico.append({"role": "user", "content": prompt_usuario})
        
        sentimento = selecionar_persona(prompt_usuario)
        if sentimento not in personas:
            sentimento = 'neutro'
        personalidade_texto = personas[sentimento]

        contexto_nome = selecionar_contexto(prompt_usuario)
        contexto_texto = selecionar_documento(contexto_nome)

        prompt_sistema = f"""
        Você é um chatbot de atendimento a clientes de um e-commerce (EcoMart). 
        Você não deve responder perguntas que não sejam dados do e-commerce informado!
        Você deve gerar respostas utilizando o contexto abaixo.
        Você deve adotar a persona abaixo.

        # Contexto
        {contexto_texto}

        # Persona
        {personalidade_texto}
        """

        mensagens = [
            {"role": "system", "content": prompt_sistema},
            *historico # Desempacota a lista de histórico aqui
        ]

        response = client.chat.completions.create(
            model=modelo,
            messages=mensagens,
            temperature=0,
            max_tokens=500
        )
        resposta_bot = response.choices[0].message["content"]
        
        # Adiciona a resposta do bot ao histórico
        historico.append({"role": "assistant", "content": resposta_bot})
        
        session["historico"] = historico
        
        return resposta_bot

    except Exception as erro:
        print("Erro no /chat:", erro)
        return f"Ocorreu um erro no servidor: {erro}", 500

@app.route("/")
def home():
    session["historico"] = []
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
