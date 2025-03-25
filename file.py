import basedosdados as bd

billing_id = "energygpt-421317"

query = """
  WITH 
dicionario_classe AS (
    SELECT
        chave AS chave_classe,
        valor AS descricao_classe
    FROM `basedosdados.br_stf_corte_aberta.dicionario`
    WHERE
        TRUE
        AND nome_coluna = 'classe'
        AND id_tabela = 'decisoes'
)
SELECT
    dados.ano as ano,
    descricao_classe AS classe,
    dados.numero as numero,
    dados.relator as relator,
    dados.link as link,
    dados.subgrupo_andamento as subgrupo_andamento,
    dados.andamento as andamento,
    dados.observacao_andamento_decisao as observacao_andamento_decisao,
    dados.modalidade_julgamento as modalidade_julgamento,
    dados.tipo_julgamento as tipo_julgamento,
    dados.meio_tramitacao as meio_tramitacao,
    dados.indicador_tramitacao as indicador_tramitacao,
    dados.assunto_processo as assunto_processo,
    dados.ramo_direito as ramo_direito,
    dados.data_autuacao as data_autuacao,
    dados.data_decisao as data_decisao,
    dados.data_baixa_processo as data_baixa_processo
FROM `basedosdados.br_stf_corte_aberta.decisoes` AS dados
LEFT JOIN `dicionario_classe`
    ON dados.classe = chave_classe
"""

df = bd.read_sql(query=query, billing_project_id=billing_id)
df.to_csv("decisoes_stf.csv", index=False)