from langchain_google_vertexai import VertexAI
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from langchain.output_parsers import ResponseSchema
import pandas as pd
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# 🔹 Configuração do modelo VertexAI
model = VertexAI(model_name="gemini-1.5-flash", project="ufg-prd-energygpt")

# 🔹 Leitura do CSV e concatenação das colunas
arquivo_csv = "/workspaces/tcc/data.csv"
df = pd.read_csv(arquivo_csv)
df['concatenado'] = df.apply(lambda row: ' | '.join(row.astype(str)), axis=1)
lista = df['concatenado'].tolist()  # Pegando apenas os primeiros 4 para teste

# 🔹 Definição do esquema de saída esperado
response_schemas = [
    ResponseSchema(name="query", description="Pergunta jurídica clara e objetiva."),
    ResponseSchema(name="resposta", description="Resumo conciso da decisão judicial."),
    ResponseSchema(name="trecho_referencia", description="Parte exata do documento que fundamenta a resposta."),
    ResponseSchema(name="número", description="Metadado do acórdão: número"),
    ResponseSchema(name="link", description="Metadado do acórdão: link"),
    ResponseSchema(name="processo", description="Metadado do acórdão: processo"),
    ResponseSchema(name="relator", description="Metadado do acórdão: relator"),
    ResponseSchema(name="data_julgamento", description="Metadado do acórdão data_julgamento."),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 🔹 Prompt formatado
prompt_template = PromptTemplate(
    input_variables=["jurisprudencia"],
    template="""
    **Instruções:**  
    Dado uma jurisprudência, gere uma query relevante para o contexto jurídico, incluindo:

    1. **Query**: Pergunta jurídica clara e objetiva.  
    2. **Resposta**: Resumo conciso da decisão judicial.  
    3. **Trecho de referência**: Parte exata do documento que fundamenta a resposta.  
    4. **Fonte**: Link para o acórdão e seus metadados (número, relator, turma, data de julgamento e publicação).  

    **Formato de saída esperado:**  
    {format_instructions}

    **Jurisprudência:**  
    {jurisprudencia}
    """
)

lock = threading.Lock()
output_list = []  # Buffer para armazenar os resultados

# 🔹 Função para processar cada item em paralelo
def process_item(idx, value):
    formatted_prompt = prompt_template.format(
        jurisprudencia=value,
        format_instructions=format_instructions
    )

    try:
        response = model.invoke(formatted_prompt)
        # print(f"Resposta do modelo ({idx}):", response)  # Debug
        parsed_output = parser.parse(response)
        
        with lock:
            output_list.append(parsed_output)  # Adiciona no buffer
        
        print(f"✔ Interação {idx} gravada no buffer.")

    except OutputParserException as e:
        print(f"❌ Erro ao processar item {idx}: {e}")

# 🔹 Processamento paralelo
try:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_item, idx, value) for idx, value in enumerate(lista)]
        for future in futures:
            future.result()  # Aguarda execução e captura erros

except Exception as e:
    print("❌ Erro geral no processamento:", e)

# 🔹 Escrevendo no JSON corretamente
with open("teste-lines.json", "w", encoding="utf-8") as json_file:
    json.dump(output_list, json_file, ensure_ascii=False, indent=4)  # Indentado para melhor visualização

print("✅ Processamento finalizado. Resultados em 'output.json'.")
