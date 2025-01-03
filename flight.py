import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
model = GenerativeModel("gemini-1.5-flash-002")

chat_session = None
if not project_id:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
    
vertexai.init(project=project_id,)

def getSession():
    global chat_session # Explicitly reference the global variable
    if not chat_session:
        chat_session = model.start_chat() # Now correctly assigns to global
    return chat_session

def getChatResponse(chat_session, prompt):
    response = chat_session.send_message(prompt)
    return response.text