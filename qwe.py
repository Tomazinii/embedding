import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_models = [
    {
        "model_name": "textmultilingualembedding002",
        "model": VertexAIEmbeddings(model_name="text-multilingual-embedding-002", project="ufg-prd-energygpt"),
        "dimension": 768
    },
    {
        "model_name": "multilinguale5large",
        "model": HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
        ),
        "dimension": 1024
    }
]









with open("/workspaces/tcc/query_synth/10lines.json", "r") as f:
    data = json.load(f).get("data")

results = []

# query = "Em um caso de tráfico de drogas, como a pena-base deve ser fixada em relação à quantidade de droga apreendida e ao uso de aplicativos para divulgação do comércio ilícito?"
# trecho_referencia = "A fixação da pena no tráfico de drogas deve considerar as circunstâncias do delito, como o uso de aplicativos para divulgação do comércio ilícito, e a quantidade de droga apreendida, sendo adequada a redução de 1/6 da pena-base em casos de apreensão de grandes quantidades de droga, como mais de 8 kg de maconha."



# doc_recuperado = """ "Paciornik e Messod Azulay Neto votaram com a Sra. Ministra Relatora.\nPresidiu o julgamento o Sr. Ministro Messod Azulay Neto.\n\n            Tese:\n            A fundamentação per relationem é válida e admitida no acórdão recorrido, não havendo negativa de prestação jurisdicional, e a dosimetria da pena deve ser feita de","""




for element in data:
    for model in embedding_models:
        model_name = model["model_name"]
        embedding_model = model["model"]

        # Calcula as embeddings
        query_embedded = embedding_model.embed_query(element["query"])
        trecho_embedded = embedding_model.embed_query(element["trecho_referencia"])
        # doc_embedded = embedding_model.embed_query(doc_recuperado)

        # Garante que as embeddings são vetores numpy
        query_vector = np.array(query_embedded).reshape(1, -1)
        trecho_vector = np.array(trecho_embedded).reshape(1, -1)
        # doc_vector = np.array(doc_embedded).reshape(1, -1)

        # Calcula similaridade cosseno
        similarity_query_reference = cosine_similarity(query_vector, trecho_vector)[0][0]
        # similarity_query_doc = cosine_similarity(query_vector, doc_vector)[0][0]

        results.append({
            "query": element["query"],
            "trecho_referencia": element["trecho_referencia"],
            "model_name": model_name,
            "similaridade_cosseno_query_reference": similarity_query_reference,
            # "similaridade_cosseno_query_doc": similarity_query_doc
        })
        
with open("qwer.json", "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)