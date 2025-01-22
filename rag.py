import os
import signal
import sys
from google.cloud import aiplatform
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings,VectorSearchVectorStore
from langchain.chains import RetrievalQA

llm_model = VertexAI(model_name="gemini-1.5-flash-002")

#You'll need to create and deploy an Index in the console, dimension 768
#https://cloud.google.com/vertex-ai/docs/vector-search/quickstart

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
if not project_id:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")

bucket_name = os.environ.get("GCS_BUCKET_NAME")
if not index_id:
    raise RuntimeError("BUCKET_NAME environment variable not set.")


embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
index_id = os.environ.get("VECTOR_SEARCH_INDEX_ID")
if not index_id:
    raise RuntimeError("VECTOR_SEARCH_INDEX_ID environment variable not set.")
index_endpoint_id = os.environ.get("VECTOR_SEARCH_INDEX_ENDPOINT_ID")
if not index_endpoint_id:
    raise RuntimeError("VECTOR_SEARCH_INDEX_ENDPOINT_ID environment variable not set.")
    
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

def load_from_directory(directory_path):
    """Loads all text files from a directory, splits them into chunks, and adds them to the vector store."""
    all_chunks = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(directory_path, filename)
            load(filepath)

# Loads a text file, splits it into chunks, and adds them to the vector store.
def load(file_name):
    print(f"-> Loading{file_name}")
    loader = TextLoader(file_name)
    documents = loader.load()

    # Initializes a RecursiveCharacterTextSplitter to split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = []
    for doc in documents:
        # Splits the document's content into chunks and adds them to the chunks list
        chunks.extend(text_splitter.split_text(doc.page_content))

    #for chunk in chunks:
    #    print(f"-->{chunk}")

    # Adds the text chunks to the vector store, overwriting any existing data
    vector_store.add_texts(texts=chunks, is_complete_overwrite=True)

def search(query):
    results = []
    results = vector_store.similarity_search(query, k=5)
    
    for result in results:
        print(f"-->{result.page_content}")

    return result

# A simple function that takes in a query and turn it into embedding with embedding_model 
def get_embedding(query):
    embeddings = embedding_model.embed_query(query)
    print(f"Embedding: {embeddings}")
    return embeddings


def recommandation(query):
    # Initialize the vectore_store as retriever
    retriever = vector_store.as_retriever()
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    # perform simple similarity search on retriever
    response = retrieval_qa.invoke({"query": f"""You are a travel agent, here you are given some context,
    base on those context, try to answer the question.

    Question: {query}
    """})

    print(f"-->{response["result"] }")

    return response



def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler

    #print(f"loading...doc...")
    #load_from_directory("./hotels")
   
    #print(f"Get embedding.... pizza.....")
    #get_embedding("pizza")
    print(f"Searching.... pizza.....")
    search("pizza")

    print(f"Get recommendation.... pizza.....")
    recommandation("I want good traditional Italian food")

    print("Program execution complete.  Press Ctrl+C to exit.")
    signal.pause()  # Keep the process running until a signal is received

