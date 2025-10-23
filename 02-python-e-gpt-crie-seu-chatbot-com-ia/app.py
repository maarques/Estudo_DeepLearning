from flask import Flask, render_template, request, Response
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

app = Flask(__name__)
app.secret_key = "alura"

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json["msg"]
    resposta = bot(prompt)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)