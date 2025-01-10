import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

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

model_config = GenerationConfig( response_mime_type="application/json", response_schema=response_schema )

def gen_better_prompts(original_prompt):
    #Initialize an array to store list of better prompt
    #better_prompts = []
    vertexai.init(project=project_id, location="us-central1")
    model=GenerativeModel("gemini-1.5-flash-002")

    prompts = [system_prompt_better_prompts]
    prompts.append(original_prompt)
    better_prompts = model._generate_content(prompts, generation_config=model_config)
    print(f"Better prompts: {better_prompts.text}")
    return better_prompts.text
