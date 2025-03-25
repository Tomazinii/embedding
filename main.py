from os import major
import os
import random
import time
from typing import List

import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from chunk_methods.chunks import ChunkDocument, ChunkFixedSize, ChunkHierarchicalParagraph, ChunkHierarchicalSection, ChunkHierarchicalSentence, ChunkSemantic, ChunkSlidingWindow, ChunkSummary
from ragas_evaluator.evaluator import evaluate_configuration
from vector_store.pgvector.pgvetor import PgvectoryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import json
import concurrent.futures
vector_store = PgvectoryVectorStore(
        db_host=os.getenv("HOST", "host.docker.internal"),
        db_port=os.getenv("PORT", 5433),
        db_user=os.getenv("USER", "myuser"),
        db_password=os.getenv("PASSWORD", "mypassword"),
        db_name=os.getenv("DATABASE", "mydatabase"),
    )
    

def embedding_chunks(embeddings_model, chunks,model_name, batch_size: int = 512):
    for i in range(0, len(chunks), batch_size):
            request = [x["content"] for x in chunks[i : i + batch_size]]
            response = embeddings_model.embed_documents(texts=request)
            for x, e in zip(chunks[i : i + batch_size], response):
                x[f"{model_name}"] = e
            print(f"Embedded {i}")
    return chunks


def insertion_vector_store(data_embed,model_names,vector_table_name):
    vector_store.indexer(documents_embedded=data_embed, model_names=model_names,vector_table_name=vector_table_name)
    
    

def chunk_orchestrator():
    pass


    
def get_best_path_by_score(query_data: dict, score_key: str = "harmonic_mean") -> dict:
    """
    Retorna o melhor path (configuração) para uma query com base no score selecionado.
    
    Args:
        query_data (dict): Estrutura contendo a query, reference_context e os paths.
        score_key (str): Score utilizado para seleção (ex: 'harmonic_mean', 'weighted_average', 'arithmetic_mean').

    Returns:
        dict: Melhor path (com config, scores, metrics, etc.)
    """
    best_path = None
    best_score = float('-inf')

    for path in query_data.get("paths", []):
        score = path.get("scores", {}).get(score_key)
        if score is not None and score > best_score:
            best_score = score
            best_path = path

    return {
        "query": query_data["query"],
        "doc_id": query_data["doc_id"],
        # "embedding": query_data["embedding"],
        "reference_context": query_data["reference_context"],
        "best_config": best_path["config"] if best_path else None,
        "best_score": best_score,
        "score_key_used": score_key,
        "metrics": best_path["metrics"] if best_path else None,
        "context_retrieved": best_path["context_retrieved"] if best_path else None,
    }
    
def search_all_methods(embedding_models, method_names, evaluator, data, top_k, similarity_threshold, table_name):
    configs=[]
    best_configs=[]
    for idx, method in enumerate(method_names):
        print(f"method model ---> {idx + 1}/{len(method_names)}")
        temp_list = []
        for idx_model, model in enumerate(embedding_models):
            print(f"embedding model ----> {idx_model + 1}/{len(embedding_models)}")
            query_embeded = model["model"].embed_query(data["query"])
            context_retrieved = vector_store.retriever(query_embedding=query_embeded,
                                                       colum_name=model["model_name"],
                                                       filter=method["chunk_method"],
                                                       top_k=top_k,
                                                       similarity_threshold=similarity_threshold,
                                                       vector_table_name=table_name,
                                                       
                                                       )
            end_antes = time.time()
            
            
            results = evaluator(
                chunk_method=method,
                embedding_model=model.get('model_name'),
                query=data["query"],
                response=data["resposta"],
                document_id=data["doc_id"],
                reference_contexts=data["trecho_referencia"],
                retrieved_contexts=context_retrieved
            )
            end_depois = time.time()
            
            
            temp_list.append(results)
        
            
        configs.extend(temp_list)
        
    # resultado = next((item for item in embedding_models if item["model_name"] == "textembeddinglargeexp0307"), None)
    # embedding = resultado["model"].embed_query(data.get("query"))
    
    output_format = {
        "query": data.get("query"),
        # "embedding": embedding,
        "doc_id": data.get("doc_id"),
        "response": data.get("resposta"),
        "reference_context": data.get("trecho_referencia"),
        "paths": configs,
    }
    
    best_result = get_best_path_by_score(
        query_data=output_format,
        score_key="harmonic_mean"
    )
    
    # vector_store.indexer_results(best_result=best_result, vector_table_name=table_name)
    
    # output_format.pop("embedding")
    
    best_configs.append(best_result)

    return output_format, best_configs
    
    
    
    
    
  
from concurrent.futures import ProcessPoolExecutor, as_completed

def training(embedding_models, query_dataset, vector_store, method_names, table_name):
    from functools import partial

    results = []
    best_results = []

    print("query_dataset length", len(query_dataset))

    # Função parcial com argumentos fixos
    func = partial(
        search_all_methods,
        embedding_models,
        method_names,
        evaluate_configuration,
        top_k=5,
        similarity_threshold=0.3,
        table_name=table_name
    )

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, data) for data in query_dataset]

        for idx, future in enumerate(as_completed(futures)):
            try:
                output, best_config = future.result()
                print(f"[{idx}] Finished query")
                best_results.append(best_config)
                results.append(output)
            except Exception as e:
                print(f"Erro ao processar query {idx}: {e}")

    final_output = {
        "results": results
    }

    final_output_best = {
        "best_results": best_results
    }

    with open("best_results.json", "w") as f:
        json.dump(final_output_best, f, ensure_ascii=False, indent=4)

    with open("results.json", "w") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
  
    
    
# def training(embedding_models, query_dataset, vector_store, method_names, table_name):
#     results = []
#     best_results = []
#     print("query_dataset length", len(query_dataset))
#     for idx, data in enumerate(query_dataset):
#         print("query: ", idx)
#         print("started query", data.get("query"))
#         output, best_config = search_all_methods(
#             embedding_models=embedding_models,
#             method_names=method_names,
#             data=data,
#             evaluator=evaluate_configuration,
#             top_k=5,
#             similarity_threshold=0.3,
#             table_name=table_name
#         )
#         best_results.append(best_config)
#         results.append(output)
        
#     final_output = {
#      "results": results   
#     }
    
#     final_output_best = {
#         "best_results": best_results
#     }
    
#     with open("best_results.json", "w") as f:
#         json.dump(final_output_best, f, ensure_ascii=False, indent=4)
    
#     with open("results.json", "w") as f:
#         json.dump(final_output, f, ensure_ascii=False, indent=4)  

    
    
    
    
def search_low_methods(embedding_models, method_names, evaluator, data, top_k,similarity_threshold):
    pass





def process_query(data, embedding_models, method_names, table_name):
    output, best_config = search_all_methods(
        embedding_models=embedding_models,
        method_names=method_names,
        data=data,
        evaluator=evaluate_configuration,
        top_k=5,
        similarity_threshold=0.3,
        table_name=table_name
    )
    return output, best_config




                


    
def process_chunk_embedding(method, model, query_dataset, model_names, iteraction_id):
    print(f"Iniciando método {method['chunk_method']} com modelo {model['model_name']}")

    chunks = method["method"].execute()
    print(f"Chunks gerados ({method['chunk_method']}): {len(chunks)}")

    chunks_embeded = embedding_chunks(
        embeddings_model=model["model"],
        chunks=chunks,
        model_name=model["model_name"]
    )

    insertion_vector_store(
        data_embed=chunks_embeded,
        model_names=model_names,
        vector_table_name=iteraction_id
    )

    print(f"Finalizado: {method['chunk_method']} + {model['model_name']}")
    
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class EmbeddingBGE:
    BASE_URL = "http://200.137.197.252:50900/embedding"  # Altere conforme necessário
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_bge_m3/query",
            json={"query": query},
            timeout=60  # você pode aumentar isso se for necessário
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_bge_m3/document",
            json={"texts": texts},
            timeout=120  # 2 minutos para batches grandes
        )
        response.raise_for_status()
        return response.json()["embeddings"]


class EmbeddingE5:
    BASE_URL = "http://200.137.197.252:50900/embedding"

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_e5/query",
            json={"query": query},
            timeout=60  # você pode aumentar isso se for necessário
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_e5/document",
            json={"texts": texts},
            timeout=120  # 2 minutos para batches grandes
        )
        response.raise_for_status()
        return response.json()["embeddings"]


class EmbeddingJINA:
    BASE_URL = "http://200.137.197.252:50900/embedding"  # Altere conforme necessário

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_jina/query",
            json={"query": query},
            timeout=60  # você pode aumentar isso se for necessário
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
        retry_if_exception_type(httpx.ReadTimeout) |
        retry_if_exception_type(httpx.HTTPStatusError)
    )
    )
    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.BASE_URL}/embedding_jina/document",
            json={"texts": texts},
            timeout=120  # 2 minutos para batches grandes
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    
    
    
def main():
    iteraction_id = "testando10"
    print("started ", iteraction_id)
    query_dataset_path = "/workspaces/tcc/query_synth/10lines.json"
    dataset_path_train = "/workspaces/tcc/data.csv"
    
    
    
    chunks_methods = [
        {
        "chunk_method":f"chunk_fixed_size_{512}",
        "method": ChunkFixedSize(dataset_path = dataset_path_train, size=512, chunk_overlap=0),
        },
        {
        "chunk_method":"full_document",
        "method": ChunkDocument(dataset_path = dataset_path_train),
        },
        {
        "chunk_method":"hierarchical_section",
        "method": ChunkHierarchicalSection(dataset_path = dataset_path_train),
        },
        {
        "chunk_method":"hierarchical_paragraph",
        "method": ChunkHierarchicalParagraph(dataset_path = dataset_path_train),
        },
        {
        "chunk_method":"hierarchical_sentence",
        "method": ChunkHierarchicalSentence(dataset_path = dataset_path_train),
        },

        {
        "chunk_method":"extractive_summary",
        "method": ChunkSummary(dataset_path = dataset_path_train),
        },
        {
        "chunk_method":"semantic_chunker",
        "method": ChunkSemantic(dataset_path = dataset_path_train),
        },
        {
        "chunk_method":"sliding_window_1024",
        "method": ChunkSlidingWindow(dataset_path = dataset_path_train),
        }
    ]

    with open(query_dataset_path, "r") as f:
        query_dataset_data = json.load(f).get("data")
        
    # for element in query_dataset_data:
    #     element["doc_id"] = element.pop("número")
        
    query_dataset = query_dataset_data
    
    
    
    # embedding_models = [
    #     {
    #         "model_name":"textmultilingualembedding002",
    #         "model": VertexAIEmbeddings(model_name="text-multilingual-embedding-002", project="ufg-prd-energygpt"),
    #         "dimension": 768
            
    #     },
    #     {
    #         "model_name":"multilinguale5large",
    #         "model": HuggingFaceEmbeddings(
    #             model_name="intfloat/multilingual-e5-large",
    #         ),
    #         "dimension": 1024
            
    #     },
    #     {
    #         "model_name":"textembeddinglargeexp0307",
    #         "model": VertexAIEmbeddings(model_name="text-embedding-large-exp-03-07", project="ufg-prd-energygpt"),
    #         "dimension": 3072
            
    #     },
    # ]
    
      
    embedding_models = [
        
        #         {
        #     "model_name":"multilinguale5large",
        #     "model": HuggingFaceEmbeddings(
        #         model_name="intfloat/multilingual-e5-large",
        #     ),
        #     "dimension": 1024
            
        # },
        {
            "model_name":"embeddingbge",
            "model": EmbeddingBGE(),
            "dimension": 1024
            
        },
        {
            "model_name":"multilinguale5large",
            "model": EmbeddingE5(),
            "dimension": 1024
            
        },
        {
            "model_name":"KaLM",
            "model": EmbeddingJINA(),
            "dimension": 896
            
        },
    ]
    
    
    model_names = [element.get("model_name") for element in embedding_models]
    # vector_store.create_table_results(
    #     dimension=3072,
    #     vector_table_name=iteraction_id
    # )
    
    vector_store.create_table(
        embedding_models=embedding_models,
        vector_table_name=iteraction_id
    )
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for method in chunks_methods:
            for model in embedding_models:
                futures.append(
                    executor.submit(
                        process_chunk_embedding,
                        method, model, query_dataset, model_names, iteraction_id
                    )
                )

        for future in concurrent.futures.as_completed(futures):
                future.result()  # Vai lançar exceção aqui se algo deu errado

    

    # training(
    #          vector_store="",
    #          embedding_models=embedding_models,
    #          query_dataset=query_dataset,
    #          method_names=chunks_methods,
    #          table_name=iteraction_id
    #          )            
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    