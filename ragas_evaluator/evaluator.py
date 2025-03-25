from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import Faithfulness
from langchain_google_vertexai import ChatVertexAI
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import LLMContextRecall


evaluator_llm = LangchainLLMWrapper(
    ChatVertexAI(
                 model_name="gemini-2.0-flash", 
                 project="ufg-prd-energygpt", 
                 location="us-central1",
                 temperature=0,
                 top_k = 1,
                 top_p=None,
                 )
)

def compute_faithfulness(query: str, response: str, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    scorer = Faithfulness(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))

# def compute_context_precision_with_reference(retrieved_contexts: list[str], reference_contexts: list[str]) -> float:
#     sample = SingleTurnSample(
#         retrieved_contexts=retrieved_contexts,
#         reference_contexts=reference_contexts
#     )
#     scorer = NonLLMContextPrecisionWithReference()
#     return float(scorer.single_turn_score(sample))



def compute_context_precision_with_reference_llm(retrieved_contexts: list[str], query: str, response) -> float:

    sample = SingleTurnSample(
        user_input=query,
        reference=response,
        retrieved_contexts=retrieved_contexts, 
    )
    scorer = LLMContextPrecisionWithReference(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))

def compute_context_precision_without_reference(query: str, response: str, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    scorer = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))

# def compute_context_recall(retrieved_contexts: list[str], reference_contexts: list[str]) -> float:
#     sample = SingleTurnSample(
#         retrieved_contexts=retrieved_contexts,
#         reference_contexts=reference_contexts
#     )
#     scorer = NonLLMContextRecall()
#     return float(scorer.single_turn_score(sample))
    
    
def compute_context_recall_with(query, reponse, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=reponse,
        reference=reponse,
        retrieved_contexts=retrieved_contexts, 
    )
    scorer = LLMContextRecall(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))
    

def weighted_average(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return sum(metrics[key] * weights.get(key, 0) for key in metrics)

def arithmetic_mean(metrics: dict[str, float]) -> float:
    return sum(metrics.values()) / len(metrics)

def harmonic_mean(metrics: dict[str, float]) -> float:
    values = list(metrics.values())
    if all(v > 0 for v in values):
        return len(values) / sum(1 / v for v in values)
    return 0.0
    
    
    
    
def evaluate_configuration(
    query: str,
    response: str,
    retrieved_contexts: list[dict],
    reference_contexts: list[str],
    embedding_model: str,
    chunk_method: str,
    document_id: str
    
) -> dict:
    if not isinstance(reference_contexts, list):
        reference_contexts = [reference_contexts]
    
    
    present_reference_document = any(str(str(document_id) in element["document_id"]) for element in retrieved_contexts)
    
    
    retrieved_contexts = [context["content"] for context in retrieved_contexts]
    
    metrics = {
        "faithfulness": compute_faithfulness(
            query=query,
            response=response,
            retrieved_contexts=retrieved_contexts),
        "context_precision_with_reference": compute_context_precision_with_reference_llm(
            response=response,
            query=query,
            retrieved_contexts=retrieved_contexts
            ),
        "context_precision_without_reference": compute_context_precision_without_reference(
            query=query,
            response=response,
            retrieved_contexts=retrieved_contexts
            ),
        "context_recall": compute_context_recall_with(
            retrieved_contexts=retrieved_contexts,
            query=query,
            reponse=response,
            )
    }

    weights = {
        "faithfulness": 0.1,
        "context_precision_with_reference": 0.3,
        "context_precision_without_reference": 0.3,
        "context_recall": 0.3,
    }

    scores = {
        "weighted_average": weighted_average(metrics, weights),
        "harmonic_mean": harmonic_mean(metrics),
        "arithmetic_mean": arithmetic_mean(metrics),
    }
    return {
        "config": {
            "embedding_model": embedding_model,
            "chunk_method": chunk_method["chunk_method"],
        },
        "metrics": metrics,
        "scores": scores,
        "context_retrieved": retrieved_contexts,
        "present_reference_document": present_reference_document
    }
    
    

# retrieved_contexts =  [
#                 "anescente, a gravidade dos crimes praticados e o histórico carcerário do apenado. III. RAZÕES DE DECIDIR\n3. No que tange à remição de pena, a aprovação no ENEM só gera direito ao benefício quando há demonstração de estudo autodidata ou atividades educacionais realizadas durante o cumprimento da pena, voltadas à conclus",
#                 "SSUAL PENAL. EXECUÇÃO PENAL. REMIÇÃO DE PENA PELA APROVAÇÃO NO ENEM. ENSINO MÉDIO CONCLUÍDO ANTES DA PRISÃO. INVIABILIDADE. VISITA PERIÓDICA AO LAR. GRAVIDADE DOS CRIMES, TEMPO REMANESCENTE DA PENA E HISTÓRICO CARCERÁRIO. INDEFERIMENTO. DECISÃO\nFUNDAMENTADA E EM CONFORMIDADE COM A LEGISLAÇÃO E JURISPRUDÊNCIA. AGRAVO CO",
#                 "dos requisitos legais. II. QUESTÃO EM DISCUSSÃO\n2. Há duas questões em discussão: (i) verificar se a aprovação no ENEM, mesmo tendo o agravante concluído o ensino médio antes da prisão, gera direito à remição de pena; (ii) analisar a viabilidade da concessão da visita periódica ao lar, considerando o tempo de pena rem",
#                 "NHECIDO E RECURSO ESPECIAL NÃO PROVIDO. I. CASO EM EXAME\n1. Agravo interposto contra decisão que inadmitiu recurso especial em que a defesa pleiteia: (i) a remição de pena pela aprovação no Exame Nacional do Ensino Médio (ENEM) e (ii) a concessão do benefício de visita periódica ao lar, sob o argumento de preenchimento",
#                 "não gera direito à remição de pena se o condenado já possuía diploma de Ensino Médio antes da prisão, e a concessão de visita periódica ao lar é inviável quando o condenado cometeu crimes de alta gravidade, possui longo histórico de faltas graves e tempo remanescente de pena superior a 23 anos."
#             ]
# reference_contexts = ["No que tange à remição de pena, a aprovação no ENEM só gera direito ao benefício quando há demonstração de estudo autodidata ou atividades educacionais realizadas durante o cumprimento da pena, voltadas à conclusão do ensino médio. O agravante, entretanto, já possuía diploma de Ensino Médio antes de sua prisão, de modo que a aprovação no exame não atende ao requisito de esforço educacional intramuros."]
# # reference_contexts = ["anescente, a gravidade dos crimes praticados e o histórico carcerário do apenado. III. RAZÕES DE DECIDIR\n3. No que tange à remição de pena, a aprovação no ENEM só gera direito ao benefício quando há demonstração de estudo autodidata ou atividades educacionais realizadas durante o cumprimento da pena, voltadas à conclus"]
# query="Em quais circunstâncias a aprovação no ENEM pode ser utilizada para a remição de pena? É possível que a aprovação no ENEM gere direito à remição de pena mesmo que o condenado tenha concluído o ensino médio antes da prisão?"
# response="A aprovação no ENEM só gera direito à remição de pena quando há demonstração de estudo autodidata ou atividades educacionais realizadas durante o cumprimento da pena, voltadas à conclusão do ensino médio. No caso em questão, o condenado já possuía diploma de Ensino Médio antes de sua prisão, de modo que a aprovação no exame não atende ao requisito de esforço educacional intramuros."








# for i in range(0,10):

#     print("compute_faithfulness(query, response, retrieved_contexts)",compute_faithfulness(
#         query=query, 
#         response=response, 
#         retrieved_contexts=retrieved_contexts
#     ))

#     print("compute_context_precision_with_reference_llm",compute_context_precision_with_reference_llm(
#                query=query, response=response, retrieved_contexts=retrieved_contexts

#     ))

#     print("compute_context_precision_without_reference",compute_context_precision_without_reference(
#                query=query, response=response, retrieved_contexts=retrieved_contexts

#     ))

#     print("compute_context_recall",compute_context_recall_with(
#         query=query,
#         reponse=response,
#         retrieved_contexts=retrieved_contexts,
        
#     ))
#     print("-------------------------------------")
       
# my_run_config = RunConfig(max_workers=15, timeout=120)
# samples= [] 
        
# for interaction_element in interactions_list:
#     sample = SingleTurnSample(
#                 user_input=interaction_element["query"],
#                 retrieved_contexts=interaction_element["retrieved_context"],
#                 response=interaction_element["generated_response"],
#                 reference=interaction_element["expected_response"],
#             )

#     samples.append(sample)

# dataset = EvaluationDataset(samples=samples)
# result = evaluate(run_config=my_run_config,dataset=dataset, metrics=metrics, llm=evaluator_llm,embeddings=embeddings_service)