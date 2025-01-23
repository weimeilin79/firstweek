import os
import signal
import sys
import vertexai
import random
from google.cloud import aiplatform
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings,VectorSearchVectorStore
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ENV SETUP
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
if not project_id:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
bucket_name = os.environ.get("GCS_BUCKET_NAME")

index_id = os.environ.get("VECTOR_SEARCH_INDEX_ID")
if not index_id:
    raise RuntimeError("VECTOR_SEARCH_INDEX_ID environment variable not set.")

index_endpoint_id = os.environ.get("VECTOR_SEARCH_INDEX_ENDPOINT_ID")
if not index_endpoint_id:
    raise RuntimeError("VECTOR_SEARCH_INDEX_ENDPOINT_ID environment variable not set.")

# Connect to resourse needed from Google Cloud
llm = VertexAI(model_name="gemini-1.5-flash-002")
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004") 
my_index = aiplatform.MatchingEngineIndex(index_id)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_id)


vector_store = VectorSearchVectorStore.from_components(
    project_id=project_id,
    gcs_bucket_name=bucket_name,
    region="us-central1",
    index_id=my_index.name,
    endpoint_id=my_index_endpoint.name,
    embedding=embedding_model,
)

mem_vector_store = InMemoryVectorStore(embedding_model)

convo_history = []

def getChatResponse(query, links):
    try:
        relevant_history = search_history(query)
        relevant_resource = search_resource(query)

        query_message = {
            "type": "text",
            "text": query,
        }

        message_parts = [query_message]
        for link in links:
            try:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": link},
                }
                message_parts.append(image_message)
            except Exception as e:
                print(f"Error processing link {link}: {e}")
                return f"Could not process image link: {link}"  
        
        

        input_msg = HumanMessage(content=message_parts)
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant that help answering question and it should be all about travel planning and advice "
                        f"Here are some past conversation history between you and the user {relevant_history}"
                        f"Here are some contenxt that is relavant to the question {relevant_resource} that you might use"
                    )
                ),
                input_msg,
            ]
        )

        prompt = prompt_template.format()
        
        response = llm.invoke(prompt)
        print(f"response: {response}")

        add_mem_store(query)
        add_mem_store(response)
        return response
    except Exception as e:
        print(f"Error sending message to chatbot: {e}") # Log this error too!
        return f"Unable to process your request at this time. Due to the following reason: {str(e)}"


def search_history(query):
    results = mem_vector_store.similarity_search(query, k=5)
    #for result in results:
    #    print(f"-->{result.page_content}")
    # Combind all the result.page_content into a single string
    combined_results = "\n".join([result.page_content for result in results])
    print(f"-->{combined_results}")

    return combined_results

def search_resource(query):
    results = []
    results = vector_store.similarity_search(query, k=5)
    
    combined_results = "\n".join([result.page_content for result in results])
    print(f"==>{combined_results}")
    return combined_results

def add_mem_store(convo):
    doc = Document(id=generate_key_id(), page_content=convo)
    mem_vector_store.add_documents(documents=[doc])


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

# Generate random key_id format as xxx-xxx-xxx 
def generate_key_id():
    return "-".join([str(random.randint(100, 999)) for _ in range(3)])

