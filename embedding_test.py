from typing import List


import httpx
from typing import List

class EmbeddingBGE:
    BASE_URL = "http://200.137.197.252:50900/embedding"  # Altere conforme necessário

    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(f"{self.BASE_URL}/embedding_bge_m3/query", json={"query": query})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(f"{self.BASE_URL}/embedding_bge_m3/document", json={"texts": texts})
        response.raise_for_status()
        return response.json()["embeddings"]


class EmbeddingE5:
    BASE_URL = "http://200.137.197.252:50900/embedding"  # Altere conforme necessário

    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(f"{self.BASE_URL}/embedding_e5/query", json={"query": query})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(f"{self.BASE_URL}/embedding_e5/document", json={"texts": texts})
        response.raise_for_status()
        return response.json()["embeddings"]


class EmbeddingJINA:
    BASE_URL = "http://200.137.197.252:50900/embedding"  # Altere conforme necessário

    def embed_query(self, query: str) -> list[float]:
        response = httpx.post(f"{self.BASE_URL}/embedding_jina/query", json={"query": query})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        response = httpx.post(f"{self.BASE_URL}/embedding_jina/document", json={"texts": texts})
        response.raise_for_status()
        return response.json()["embeddings"]


# bge = EmbeddingE5()
# result = bge.embed_query("exemplo de query asdad sad sasd exemplo de query asdad sad sasdexemplo de query asdad sad sasdexemplo de query asdad sad sasdexemplo de query asdad sad sasd")
# print(len(result))

# docs_result = bge.embed_documents(["doc1", "doc2"])
# print(docs_result)