import os
import json
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
from flight import *
from evaluation import *


app = Flask(__name__)

@app.route('/')
def index():
    """Return a Index of available test pages."""
    return render_template('index.html')

#function that will route to test.html
@app.route('/test')
def test():
    return render_template('test.html')

#function that will route to vertex.html
@app.route('/test-vertex')
def testVertex():
    return render_template('test-vertex.html')

#function that will route to flight.html
@app.route('/flight')
def testFlight():
    return render_template('flight.html')

#function that will route to evaluation.html
@app.route('/evaluation')
def testEvaluation():
    return render_template('evaluation.html')

#function that handles the request(POST) that receives the prompt from test.html and will call the Gemma model in vertex   
@app.route('/call-gemma', methods=['POST'])
def handle_prompt():
    data = request.get_json()
    prompt = data.get('prompt')
    token = data.get('token')
    #Call Gemma model in Vertex AI here.  
    response = call_gemma_model(prompt,token)

    return response

@app.route('/call-vertex', methods=['POST'])
def handle_prompt_vertex():
    data = request.get_json()
    prompt = data.get('prompt')

    #Call Gemma model in Vertex AI here.  
    response = call_vertex(prompt)
    return response

@app.route('/chatbot', methods=['POST'])
def handle_chatbot():
    data = request.get_json()
    message = data.get('message')
    links = data.get('links', []) # Get links, default to empty list if not provided
    print(f"Received links: {links}")  

    chat_session = getSession()
    chat_response = getChatResponse(chat_session, message, links)
    print(f"Sending response: {chat_response}") # Debug print statement
    response = {"response": chat_response} # Changed key to 'response'
    return jsonify(response)
    
@app.route('/call-evaluation', methods=['POST'])
def handle_evaluation(): 
    data = request.get_json()
    prompt = data.get('prompt')
    print(f"{prompt}")
    response_str = gen_better_prompts(prompt)  
    response_list = json.loads(response_str) # Parses string to list of dict
    return jsonify(response_list) # jsonify the list

@app.route('/evaluate_genai', methods=['POST'])
def handle_evaluation_genai(): 
    data = request.get_json()
    prompt = data.get('prompt')
    print(f"{prompt}")
    response = evaluate_genai(prompt)
    return response

#Implement the call_gemma_model using vertex sdk

def call_gemma_model(prompt,token):
    genai.configure(api_key=token)
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    response = model.generate_content(prompt)
    return response.text

def call_vertex(prompt):
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
    vertexai.init(project=project_id,)

    # Tunned model 
    # projects/pure-album-446616-n3/locations/us-central1/models/108061102289453056
    # Model
    # gemini-1.5-flash-002
    model = GenerativeModel("projects/pure-album-446616-n3/locations/us-central1/endpoints/108061102289453056")
    response = model.generate_content(prompt)
    return response.text




if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')

