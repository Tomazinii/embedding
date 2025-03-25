


import json
from langchain_google_vertexai import ChatVertexAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.metrics import LLMContextRecall
from ragas.llms import LangchainLLMWrapper


my_run_config = RunConfig(max_workers=15, timeout=120)

# evaluator_llm= ChatGoogleGenerativeAI(
#             model="gemini-1.5-pro",
#         )



metrics = [LLMContextRecall(), LLMContextPrecisionWithReference(), LLMContextPrecisionWithoutReference(), Faithfulness()]


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

with open("/workspaces/tcc/results.json", "r") as file:
    data = json.load(file).get("results")
    
    

for element in data:
    reference_context = element["reference_context"]
    query = element["query"]
    response = element["response"]
    samples = []
    for path in element["paths"]:
        sample = SingleTurnSample(
                        user_input=query,
                        retrieved_contexts=path["context_retrieved"],
                        response=response,
                        reference=response,
                    )
        samples.append(sample)


        
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(run_config=my_run_config,dataset=dataset, metrics=metrics, llm=evaluator_llm,batch_size=10)

    df_resultados = result.to_pandas().to_dict(orient='list')
    result_dict = {
                    'faithfulness': df_resultados['faithfulness'],
                    'llm_context_precision_with_reference': df_resultados['llm_context_precision_with_reference'],
                    'llm_context_precision_without_reference':df_resultados['llm_context_precision_without_reference'],
                    'context_recall': df_resultados['context_recall']  
            }
    print(df_resultados)
