# Flight Assist: A Vertex AI and GCP Exploration

This repository showcases a self-learning project exploring Vertex AI and other GCP technologies.  It demonstrates several key functionalities, starting with basic tests and culminating in a multimodal flight assistant application with integrated evaluation capabilities.

## Project Components

* **Basic Gemma Tests:**  Includes simple test applications demonstrating how to call the Gemini "flash" models both via the Generative AI Studio and directly through the Vertex AI platform.  These tests provide a foundation for understanding basic model interaction. (`/test`, `/test-vertex`)

* **Flight Assistant Chatbot:** This application simulates a flight booking assistant.  It uses a Gemini model to answer user queries related to flights, destinations, and trip planning.  It supports both text-based conversations and multimodal inputs, allowing users to submit images (via URLs) as part of their requests. (`/flight`)

* **Prompt Evaluation:** This component demonstrates how to evaluate different Large Language Models (LLMs) during inference time. It uses pointwise evaluation metrics such as coherence, verbosity, and question-answering quality to compare the responses of different models for a given prompt. This section also allows exploring better prompts. (`/evaluation`)

* **RAG Integration:**  Uses Vector Search to retrieve related doc with embedding


## Key Technologies Used

* **Vertex AI:** Core platform for deploying and interacting with LLMs.
* **Gemini:** Family of large language models used throughout the project.
* **Generative AI Studio:**  Used for initial model testing and exploration.
* **Flask:**  Python web framework used to build the user interface.


## Setup and Installation

1. **Clone the repository:**  `git clone ...`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Set environment variables:**
    * `GOOGLE_CLOUD_PROJECT`: Your GCP project ID.
4. **Run the Flask application:** `python3 app.py`


For RAG
1. **Run the application:** `python3 rag.py`

## Future Work
* Security 
* Reasoning 
* Agentic
