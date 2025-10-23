from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model= modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

def category_product(product_name, list_possible_categories):
    prompt_system = f"""
        Você é um categorizador de produtos.
        Você deve assumir as categorias presentes na lista abaixo.

        # Lista de Categorias Válidas
            {list_possible_categories.split(",")}

        # Formato da Saída
            Produto: Nome do Produto
            Categoria: apresente a categoria do produto

        # Exemplo de Saída
            Produto: Escova elétrica com recarga solar
            Categoria: Eletrônicos Verdes
    """

    output = client.chat.completions.create(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": product_name
            }
        ],
        max_tokens=1024,
        temperature=0,
    )

    return output.choices[0].message["content"]

valid_categories = input("Informe as categorias válidas, separando por vígula: ")

while True:
    product_name = input("Digite o nome do produto: ")
    output_text = category_product(product_name, valid_categories)
    print(output_text)
