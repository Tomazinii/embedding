# Agrupar os dados por tipo de pedido e contar a quantidade de processos por categoria
import pandas as pd
import ace_tools as tools

# Carregar o arquivo CSV
file_path = "/workspaces/tcc/jurisprudencias_stj.csv"
df = pd.read_csv(file_path)

df.head()
df_grouped = df.groupby("descricao_pedido").size().reset_index(name="quantidade")

# Exibir os dados para o usuário
tools.display_dataframe_to_user(name="Processos por Tipo de Pedido", dataframe=df_grouped)
