import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.evaluation import EvalTask,MetricPromptTemplateExamples, PointwiseMetricPromptTemplate, PointwiseMetric, PairwiseMetric
import pandas as pd  # Make sure pandas is imported


project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
if not project_id:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
system_prompt_better_prompts ="""
You are an AI assistant that will faciliate and come up with 3 better prompt options for the following original question

original question: 
"""
# Documenation on controled output
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output

#Define the response schema

response_schema = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "response": {
        "type": "string"
      }
    },
    "required": [
      "response"
    ]
  }
}

vertexai.init(project=project_id, location="us-central1")
model_config = GenerationConfig( response_mime_type="application/json", response_schema=response_schema )

def gen_better_prompts(original_prompt):
    #Initialize an array to store list of better prompt
    #better_prompts = []
    model=GenerativeModel("gemini-1.5-flash-002")

    prompts = [system_prompt_better_prompts]
    prompts.append(original_prompt)
    better_prompts = model._generate_content(prompts, generation_config=model_config)
    print(f"Better prompts: {better_prompts.text}")
    return str(better_prompts.text)

# A great reference for this inference evaluation
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_models_in_vertex_ai_studio_and_model_garden.ipynb
# Awesome slides
# https://docs.google.com/presentation/d/1F9YM_qFNWLg2VMamYFyRWHFQvXtASlMhl0sz-Aav7mg

def evaluate_genai(prompt):
    gemini_flash = GenerativeModel("gemini-1.5-flash-002")
    llama_model = GenerativeModel("projects/967543736643/locations/us-central1/endpoints/8951213222166265856")

    results = []  # Now a list of dictionaries

    results.append({"input": str(prompt), "output": str(gemini_flash.generate_content(prompt).text)})
    results.append({"input": str(prompt), "output": str(llama_model.generate_content(prompt).text)})

    print(f"Results: {results}")

    # Convert the dict to Panda Data Frame
    df = pd.DataFrame(results)
    
    return run_eval_task(df)



def run_eval_task( lmm_results):
  model=GenerativeModel("gemini-1.5-flash-002")

  print("Just FYI: Here are your evaluation options out of box-->")
  example_metric_names = MetricPromptTemplateExamples.list_example_metric_names()
  for metric_name in example_metric_names:
    print(metric_name)

  pointwise_eval_task = EvalTask(
    dataset=lmm_results,
    metrics=["coherence","verbosity","question_answering_quality","summarization_quality"],
  )

  pointwise_result = pointwise_eval_task.evaluate(
    model=model,
    prompt_template="# Input\n{input} # Output\n{output}",
  )

  print(f"Pointwise_result: {pointwise_result}")
  print("---------------------")
  print(type(pointwise_result))  # Print the object's type
  print("---------------------")
  print(dir(pointwise_result))   # Print the object's attributes

  return util_turnEvalResult2json(pointwise_result)



def util_turnEvalResult2json(result):
    json_result = {} 
    json_result["summary_metrics"] = result.summary_metrics # Add summary metrics
    json_result["metrics_table"] = result.metrics_table.to_dict(orient='records')


    json_str = json.dumps(json_result, indent=2, default=str) #default=str to handle numpy data types
    print(json_str)
    return json_str # Returns the proper json string.