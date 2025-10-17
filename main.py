from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    model= "meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

output = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "Listar apenas os nomes dos produtos, sem considerar a descrição."
        },
        {
            "role": "user",
            "content": "Liste 3 produtos sustentáveis."
        }
    ],
    max_tokens=1024
)

print(output.choices[0].message["content"])