# First the training data to the cloud storage, 
# https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning#python
# gcloud storage buckets create gs://446616-training-data

import os
import vertexai
from vertexai.tuning import sft

vertexai.init(PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"])

sft_tuning_job = sft.train(
    source_model="gemini-1.5-flash-002",
    train_dataset="gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl" #input and output

)

while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()


print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)