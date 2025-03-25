import json
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extras import execute_values





class PgvectoryVectorStore:
    
    def __init__(self ,db_name: str, db_port: int, db_user: str, db_password: str, db_host: str) -> None:
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        
    def _connect(self):
        """Estabelece a conexão com o banco de dados PostgreSQL."""
        return psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        
    def retriever(self, query_embedding: list, colum_name: str, similarity_threshold, top_k, vector_table_name, filter):
        """
        Recupera conteúdos do banco de dados com base na similaridade do embedding.

        :param query_embedding: Vetor de consulta (embedding).
        :param similarity_threshold: Limiar mínimo de similaridade (float entre 0 e 1).
        :param num_matches: Número máximo de correspondências desejadas.
        :return: Lista de dicionários contendo os conteúdos correspondentes.
        """
        
        filter_clause = ""
        filter_params = ()

        if filter:
            filter_clause = "AND chunk_type = %s"
            filter_params = (filter,)

        query = f"""
        WITH vector_matches AS (
            SELECT 
                content, document_id, chunk_type, {colum_name},
                1 - ({colum_name} <=> %s::vector) AS similarity
            FROM {vector_table_name}
            WHERE 1 - ({colum_name} <=> %s::vector) > %s
            {filter_clause}
            ORDER BY similarity DESC
            LIMIT %s
        )
        SELECT content, document_id, chunk_type, {colum_name}, similarity FROM vector_matches
        """

        conn = None
        matches = []

        try:
            conn = self._connect()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Montar parâmetros dinamicamente
            params = (query_embedding, query_embedding, similarity_threshold) + filter_params + (top_k,)

            cursor.execute(query, params)
            results = cursor.fetchall()

            if len(results) == 0:
                raise Exception("Did not find any results. Adjust the query parameters.")

            for r in results:
                chunk = {
                    "content": r.get("content"),
                    "document_id": r.get("document_id"),
                    "chunk_type": r.get("chunk_type"),
                    "similarity_with_query": r.get("similarity"),
                    # f"{colum_name}": r.get(f"{colum_name}")
                }
                matches.append(chunk)

        except Exception as e:
            print(f"Erro ao recuperar documentos: {e}")

        finally:
            if conn:
                cursor.close()
                conn.close()

        return matches
    
    
    def retriever_results(self, query_embedding: list, colum_name: str, similarity_threshold, top_k, vector_table_name, filter):
        """
        Recupera conteúdos do banco de dados com base na similaridade do embedding.

        :param query_embedding: Vetor de consulta (embedding).
        :param similarity_threshold: Limiar mínimo de similaridade (float entre 0 e 1).
        :param num_matches: Número máximo de correspondências desejadas.
        :return: Lista de dicionários contendo os conteúdos correspondentes.
        """

        
        query = f"""
        WITH vector_matches AS (
            SELECT 
                query, best_score, best_config, document_id, {colum_name},
                1 - ({colum_name} <=> %s::vector) AS similarity
            FROM {vector_table_name}_search
            WHERE 1 - ({colum_name} <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s
        )
        SELECT query, document_id, best_score,best_config, {colum_name}, similarity FROM vector_matches
        """

        conn = None
        matches = []

        try:
            conn = self._connect()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Executar consulta com parâmetros
            cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, top_k))
            # Obter resultados
            results = cursor.fetchall()
            
            if len(results) == 0:
                raise Exception("Did not find any results. Adjust the query parameters.")
            # Construir a lista de correspondências
            for r in results:
                chunk = {
                    "query": r.get("query"),
                    "document_id": r.get("document_id"),
                    # "best_score": r.get("best_score"),
                    "similarity_with_query": r.get("similarity"),
                    "best_config": r.get("best_config"),
                }
                matches.append(chunk)
        
        except Exception as e:
            print(f"Erro ao recuperar documentos: {e}")
        
        finally:
            # Fechar conexão apenas se ela foi criada
            if conn:
                cursor.close()
                conn.close()

        return matches
    
    
    def create_table_results(self, vector_table_name: str, dimension:list):
        """
        Cria uma nova tabela PostgreSQL para armazenar documentos e embeddings.
        """


        
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {vector_table_name}_search (
                query TEXT NOT NULL,
                document_id VARCHAR(255),
                reference_context TEXT,
                best_config JSONB,
                embedding vector({dimension}),
                best_score FLOAT,
                score_key_used TEXT,
                metrics JSONB,
                context_retrieved JSONB
                
            );
            """
        
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(f"DROP TABLE IF EXISTS {vector_table_name}_search")
        print(f"Tabela {vector_table_name} removida")
        
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        
        print("Tabela criada com sucesso.")
    
    def indexer_results(self, best_result, vector_table_name):
            
            

            insert_query = f"""
            INSERT INTO {vector_table_name}_search (
                query,
                document_id,
                embedding,
                reference_context,
                best_config,
                best_score,
                score_key_used,
                metrics,
                context_retrieved
                )
            VALUES %s
            """
            
        
            values = []
            if best_result:
                row = [
                    best_result.get("query"),
                    best_result.get("doc_id"),
                    best_result.get("embedding"),
                    best_result.get("reference_context"),
                    json.dumps(best_result.get("best_config")),
                    best_result.get("best_score"),
                    best_result.get("score_key_used"),
                    json.dumps(best_result.get("metrics")),
                    json.dumps(best_result.get("context_retrieved"))
                ]
                values.append(tuple(row))
                    
    
            try:
                conn = self._connect()
                
                cursor = conn.cursor()
                
                execute_values(cursor, insert_query, values)
                
                conn.commit()
                print(f"registros inseridos com sucesso na tabela {vector_table_name}.")
                cursor.close()
                conn.close()
                
                
            except Exception as e:
                print(f"Erro ao inserir documentos: {e}")
            finally:
                if conn:
                    cursor.close()
                    conn.close()
    
    def indexer(self, documents_embedded: list[dict], model_names: list[str], vector_table_name):
        
        columns = []
        for name in model_names:
            columns_definition = f"{name}"
            columns.append(columns_definition)

        columns_str = ",\n            ".join(columns)

        
        insert_query = f"""
        INSERT INTO {vector_table_name} (
            content,
            document_id,
            chunk_type,
            {columns_str}
            )
        VALUES %s
        """
    
        values = []
        if documents_embedded:
            for doc in documents_embedded:
                row = [
                    doc.get("content"),
                    doc.get("doc_id"),
                    doc.get("chunk_type")
                ]
                for name in model_names:
                    row.append(doc.get(name))
                    
                values.append(tuple(row))
                
        try:
            conn = self._connect()
            
            cursor = conn.cursor()
            
            execute_values(cursor, insert_query, values)
            
            conn.commit()
            print(f"registros inseridos com sucesso na tabela {vector_table_name}.")
            cursor.close()
            conn.close()
            
            
        except Exception as e:
            print(f"Erro ao inserir documentos: {e}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    def create_table(self, vector_table_name: str, embedding_models:list):
        """
        Cria uma nova tabela PostgreSQL para armazenar documentos e embeddings.
        """
   
        columns = []
        for model in embedding_models:
            columns_definition = f"{model["model_name"]} vector({model["dimension"]})"
            columns.append(columns_definition)

        columns_str = ",\n            ".join(columns)

        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {vector_table_name} (
                content TEXT NOT NULL,
                document_id VARCHAR(255),
                chunk_type TEXT NOT NULL,
                {columns_str}
            );
            """
        
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {vector_table_name}")
        
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        
        print("Tabela criada com sucesso.")
        


    def remove_embeddings(self, document_id: str):
        """
        Remove embeddings associados a um document_id específico.

        :param document_id: O ID do documento cujos embeddings serão removidos.
        """
        delete_query = f"""
        DELETE FROM {self.vector_table_name}
        WHERE document_id = %s
        """

        conn = None

        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Executar a consulta de exclusão
            cursor.execute(delete_query, (document_id,))
            
            # Confirmar a exclusão
            conn.commit()

        except Exception as e:
            print(f"Erro ao remover embeddings: {e}")
        
        finally:
            # Fechar a conexão apenas se ela foi criada
            if conn:
                cursor.close()
                conn.close()
