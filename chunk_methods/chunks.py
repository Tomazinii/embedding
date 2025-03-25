import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

import pandas as pd

from .chunk_functions import extractive_summary, hierarchical_chunking, markdown_chunker, semantic_chunker, sliding_window_chunker

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)

    documentos = []
    

    for _, row in df.iterrows():
        conteudo = f"""
        
            # Dados do processo
            Número do Processo: {row['numero_processo']}
            Descrição do Pedido: {row['descricao_pedido']}
            Número do Registro: {row['numero_registro']}
            Documento Atual/Total: {row['numero_doc_atual']}/{row['numero_doc_total']}
            Identificação: {row['identificacao']}
            Relator: {row['relator']}
            Data de Julgamento: {row['data_julgamento']}
            Órgão Julgador: {row['orgao_julgador']}
            
            # Ementa: 
            {row['ementa']}

            # Acordão:
            {row['acordao_text']}

            # Tese:
            {row['tese']}
        """.strip()

        documentos.append({
            "content": conteudo,
            "doc_id": row['numero_processo']
        })

    return documentos





class ChunkDocument:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        for document in documents:
            document["chunk_type"] = "full_document" 
        return documents




class ChunkHierarchicalSection:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            
            texts: list[str] = hierarchical_chunking(
                text=document["content"],
                levels=["section"],
                doc_id=document["doc_id"]
            )["sections"]
            chunks.extend(texts)
        return chunks



class ChunkHierarchicalParagraph:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            texts: list[str] = hierarchical_chunking(
                text=document["content"],
                levels=["paragraph"],
                doc_id=document["doc_id"]
            )["paragraphs"]
            chunks.extend(texts)
        return chunks





class ChunkHierarchicalSentence:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            texts: list[str] = hierarchical_chunking(
                text=document["content"],
                levels=["sentence"],
                doc_id=document["doc_id"]
            )['sentences']
            chunks.extend(texts)
        return chunks





class ChunkSummary:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            text = extractive_summary(
                text=document["content"],
                limit=512
            )
            chunk = {"content": text, "chunk_type":"extractive_summary", "doc_id": document["doc_id"]}
            chunks.append(chunk) 
        return chunks
    
    
    
    
    
class ChunkSemantic:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        
        chunks = []
        
        for document in documents:
            texts = semantic_chunker(
                text=document["content"],
                embedding_model=VertexAIEmbeddings(model_name="text-multilingual-embedding-002", project="ufg-prd-energygpt")
            )
            chunk = [{"content": text, "chunk_type": "semantic_chunker", "doc_id":document["doc_id"]} for text in texts] 
            chunks.extend(chunk)
            
        return chunks


    
class ChunkSlidingWindow:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            texts = sliding_window_chunker(
                text=document["content"],
                chunk_size=1024,
                
            )
            chunk = [{"content": text, "chunk_type": "sliding_window_1024", "doc_id":document["doc_id"]} for text in texts]
            
            chunks.extend(chunk)
        return chunks

    
class ChunkMarkdown:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def execute(self):
        documents = load_data(self.dataset_path)
        chunks = []
        for document in documents:
            texts = markdown_chunker(
                text=document["content"]
            )
            chunk = [{"content": text, "chunk_type": "markdown_chunker", "doc_id": document["doc_id"]} for text in texts]
            chunks.extend(chunk)
            
        return chunks



class ChunkFixedSize:
    
    def __init__(self, dataset_path, size, chunk_overlap):
        self.dataset_path = dataset_path
        self.size = size
        self.chunk_overlap = chunk_overlap
        
    def execute(self):
        documents = load_data(self.dataset_path)
        
        chunks = []
        for document in documents:
            
            text_splitter = CharacterTextSplitter(
                separator="",  
                chunk_size=self.size,
                chunk_overlap=self.chunk_overlap
            )
            
            texts = text_splitter.split_text(document["content"])
            chunk_add_info = [{"content": text, "chunk_type": f"chunk_fixed_size_{self.size}", "doc_id":document["doc_id"]} for text in texts]
            chunks.extend(chunk_add_info)
            
        return chunks



