from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json

load_dotenv()

modelo = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    model=modelo,
    api_key=os.getenv("HUGGINGFACE_TOKEN_KEY")
)

def carrega(nome_do_arquivo):
    
    try:
        with open(nome_do_arquivo, "r", encoding="utf-8") as arquivo:
            return arquivo.read()
    except IOError as e:
        print(f"Erro: {e}")

def salva(nome_do_arquivo, conteudo):
    
    try:
        with open(nome_do_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
    except IOError as e:
        print(f"Erro ao salvar arquivo: {e}")

def analisar_transacao(lista_transacoes):
    
    print("1.Executando a análise de transação")
    
    prompt_sistema = """
    Analise as transações financeiras a seguir e identifique se cada uma delas é uma "Possível Fraude" ou deve ser "Aprovada". 
    Adicione um atributo "Status" com um dos valores: "Possível Fraude" ou "Aprovado".

    Cada nova transação deve ser inserida dentro da lista do JSON.

    # Possíveis indicações de fraude
    - Transações com valores muito discrepantes
    - Transações que ocorrem em locais muito distantes um do outro
    
    Não usar blocos de código (sem ``` e sem texto explicativo).
    Adote o formato de resposta abaixo para compor sua resposta.
        
    # Formato Saída 
    {
        "transacoes": [
            {
            "id": "id",
            "tipo": "crédito ou débito",
            "estabelecimento": "nome do estabelecimento",
            "horário": "horário da transação",
            "valor": "R$XX,XX",
            "nome_produto": "nome do produto",
            "localização": "cidade - estado (País)"
            "status": ""
            },
        ]
    } 
    """

    lista_mensagens = [
        {
                "role": "system",
                "content": prompt_sistema
        },
        {
                "role": "user",
                "content": f"""Considere o CSV abaixo, onde cada linha é uma 
                transação diferente: {lista_transacoes}. Sua resposta deve adotar o 
                #Formato de Resposta (apenas um json sem outros comentários)"""
        }
    ]

    resposta = client.chat.completions.create(
        messages=lista_mensagens,
        model=modelo,
        temperature=0
    )

    conteudo = resposta.choices[0].message["content"]
    print("\Conteúdo:", conteudo)
    json_resultado = json.loads(conteudo)
    print("\nJSON:", json_resultado)
    return json_resultado

lista_transacoes = carrega("dados/transacoes.csv")
transacoes_analisadas = analisar_transacao(lista_transacoes)

def gerar_parecer(transacao):
    
    print("2.Gerando um parecer para cada transação")

    prompt_sistema = f"""
    Para a seguinte transação, forneça um parecer, apenas se o status dela for de "Possível Fraude". Indique no parecer uma justificativa para que você identifique uma fraude.
    Transação: {transacao}

    ## Formato de Resposta
    "id": "id",
    "tipo": "crédito ou débito",
    "estabelecimento": "nome do estabelecimento",
    "horario": "horário da transação",
    "valor": "R$XX,XX",
    "nome_produto": "nome do produto",
    "localizacao": "cidade - estado (País)"
    "status": "",
    "parecer" : "Colocar Não Aplicável se o status for Aprovado"
    """

    lista_mensagens = [
        {
            "role": "user",
            "content": prompt_sistema
        }
    ]

    resposta = client.chat.completions.create(
        messages = lista_mensagens,
        model=modelo,
    )

    conteudo = resposta.choices[0].message["content"]
    print("Finalizou a geração do parecer")
    return conteudo

def gerar_recomendacao(parecer):
    
    print("3.Gerando recomendações")

    prompt_sistema = f"""
    Para a seguinte transação, forneça uma recomendação apropriada baseada no status e nos detalhes da transação da Transação: {parecer}

    As recomendações podem ser "Notificar Cliente", "Acionar setor Anti-Fraude" ou "Realizar Verificação Manual".
    Elas devem ser escritas no formato técnico.

    Inclua também uma classificação do tipo de fraude, se aplicável. 
    """

    lista_mensagens = [
        {
                "role": "system",
                "content": prompt_sistema
        }
    ]

    resposta = client.chat.completions.create(
            messages = lista_mensagens,
            model=modelo,
    )

    conteudo = resposta.choices[0].message["content"]
    print("Finalizou geração de recomendações")
    return conteudo

for transacoes in transacoes_analisadas["transacoes"]:
    if transacoes["status"] == "Possível Fraude":
        parecer = gerar_parecer(transacoes)
        recomendacao = gerar_recomendacao(parecer)
        id_transacao = transacoes["id"]
        produto_transacao = transacoes["nome_produto"]
        status_transacao = transacoes["status"]
        salva(f"transacao-{id_transacao}-{produto_transacao}-{status_transacao}.txt", recomendacao)
