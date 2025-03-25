from langchain_huggingface import HuggingFaceEmbeddings

model_name = "intfloat/multilingual-e5-large"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    # model_kwargs=model_kwargs,
    # encode_kwargs=encode_kwargs
)

query = "Em quais situações a aprovação no ENEM pode ser utilizada para remição de pena em regime de cumprimento de pena privativa de liberdade?"

print(hf.embed_query(query))