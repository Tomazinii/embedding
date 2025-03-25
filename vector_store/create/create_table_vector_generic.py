
import psycopg2
import os

# Dados de conexão ao banco de dados
DB_NAME = os.getenv("POSTGRES_DB", "mydatabase")
DB_USER = os.getenv("POSTGRES_USER", "myuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")
DB_HOST = os.getenv("POSTGRES_HOST", "host.docker.internal")
DB_PORT = os.getenv("POSTGRES_PORT", 5433)


def create_table(table_name: str, embedding_model_info: list):
    DB_TABLE_NAME = table_name

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cur = conn.cursor()

    # drop_table = f"""DROP TABLE IF EXISTS {table_name}"""
    # cur.execute(drop_table

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME}")

    
    columns = []
    for element in embedding_model_info:
        columns_definition = f"{element["embedding_name"]} vector({element["dimension"]})"
        columns.append(columns_definition)

    columns_str = ",\n            ".join(columns)

    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            content TEXT NOT NULL,
            metada JSONB,
            {columns_str}
        );
        """
        


    cur.execute(create_table_query)

    # Confirma as alterações no banco
    conn.commit()

    # Fecha o cursor e a conexão
    cur.close()
    conn.close()

    print(f"Tabela '{table_name}' criada com sucesso.")

    
    
    
lista = [
        {"embedding_name": "a", "dimension":768},
        {"embedding_name": "b", "dimension":768},
        {"embedding_name": "c", "dimension":768},
        {"embedding_name": "d", "dimension":768},
        ]
create_table(embedding_model_info=lista, table_name="embeddings")
print(f"Tabela criada com sucesso.")
    