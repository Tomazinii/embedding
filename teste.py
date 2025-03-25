# import json
# import os

# from langchain_google_vertexai import VertexAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from vector_store.pgvector.pgvetor import PgvectoryVectorStore


# vector_store = PgvectoryVectorStore(
#         db_host=os.getenv("HOST", "host.docker.internal"),
#         db_port=os.getenv("PORT", 5433),
#         db_user=os.getenv("USER", "myuser"),
#         db_password=os.getenv("PASSWORD", "mypassword"),
#         db_name=os.getenv("DATABASE", "mydatabase"),
#     )
    
    
        
# import json
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# embedding_models = [
#     {
#         "model_name": "textmultilingualembedding002",
#         "model": VertexAIEmbeddings(model_name="text-multilingual-embedding-002", project="ufg-prd-energygpt"),
#         "dimension": 768
#     },
#     {
#         "model_name": "multilinguale5large",
#         "model": HuggingFaceEmbeddings(
#             model_name="intfloat/multilingual-e5-large",
#         ),
#         "dimension": 1024
#     },
#     {
#         "model_name": "textembeddinglargeexp0307",
#         "model": VertexAIEmbeddings(model_name="text-embedding-large-exp-03-07", project="ufg-prd-energygpt"),
#         "dimension": 3072
#     },
# ]

# with open("/workspaces/tcc/query_synth/teste-lines.json", "r") as f:
#     data = json.load(f).get("data")

# results = []

# for element in data:
#     query = element["query"]
#     trecho_referencia = element["trecho_referencia"]

#     for model in embedding_models:
#         model_name = model["model_name"]
#         embedding_model = model["model"]

#         # Calcula as embeddings
#         query_embedded = embedding_model.embed_query(query)
#         trecho_embedded = embedding_model.embed_query(trecho_referencia)

#         # Garante que as embeddings são vetores numpy
#         query_vector = np.array(query_embedded).reshape(1, -1)
#         trecho_vector = np.array(trecho_embedded).reshape(1, -1)

#         # Calcula similaridade cosseno
#         similarity = cosine_similarity(query_vector, trecho_vector)[0][0]

#         results.append({
#             "query": query,
#             "trecho_referencia": trecho_referencia,
#             "model_name": model_name,
#             "similaridade_cosseno": similarity
#         })
# print(results)

    #     results = vector_store.retriever_results(
    #         colum_name="embedding",
    #         filter="",
    #         query_embedding=query_embeded,
    #         similarity_threshold=0.3,
    #         top_k=1,
    #         vector_table_name=iteraction_id
    #         )

    # print(f"query: {query} \n doc_id:{results[0]["document_id"]} \n -------> best config: {results[0]["best_config"]}")
    # print("---------------------------------------------------------")


# import pandas as pd



# df = pd.read_csv("/workspaces/tcc/teste.csv")
# columns = ["content", "doc_id"]
# documents = df[columns].to_dict(orient="records")

# print(documents)



# from main import EmbeddingBGE


from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# embedding_model = VertexAIEmbeddings(model_name="text-multilingual-embedding-002", project="ufg-prd-energygpt")
embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                # encode_kwargs={"batch_size": 256}
            )



def embedding_chunks(embeddings_model, chunks,model_name, batch_size: int = 3):
    for i in range(0, len(chunks), batch_size):
            request = [x["content"] for x in chunks[i : i + batch_size]]
            response = embeddings_model.embed_documents(texts=request)
            for x, e in zip(chunks[i : i + batch_size], response):
                x[f"{model_name}"] = e
            print(f"Embedded {i}")
    return chunks




chunks = [
    {"content":"texto1"},
    {"content":"texto2"},
    {"content":"texto3"},
    {"content":"texto4"},
    {"content":"texto5"},
    {"content":"texto6"},

          ]

output = chunks_embeded = embedding_chunks(
    embeddings_model=embedding_model,
    chunks=chunks,
    model_name="embedding_model"
)

print(len(output))