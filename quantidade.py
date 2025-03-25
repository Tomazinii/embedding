import pandas as pd

# Carregar o CSV
arquivo_csv = "/workspaces/tcc/Processos_por_Tipo_de_Pedido.csv"  # Substitua pelo caminho correto do arquivo

df = pd.read_csv(arquivo_csv)

# Filtrar os pedidos com quantidade maior que 100
df_filtrado = df[df['quantidade'] > 100]

# Exibir o resultado
print(df_filtrado['quantidade'].sum())

# Opcional: Salvar o resultado em um novo arquivo CSV
# df_filtrado.to_csv("pedidos_abaixo_100.csv", index=False)
