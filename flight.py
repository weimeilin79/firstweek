import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Part

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
        system_prompt = """
        You are a consultant of a traveling agengcy, the client will ask you question about flights, destination and anything related to planning a trip. 
        Don't answer if the question is not related to traveling, or trip planning. Politely ask client to not to derail. 
        """
        response = chat_session.send_message(system_prompt)
    return chat_session

def getChatResponse(chat_session, prompt, links):
    prompts = []
    for link in links:
        prompts.append(
            Part.from_uri(link, "image/jpeg")
        )  # Assumes all links are valid URIs
    try:
        prompts.append(prompt)
        response = chat_session.send_message(prompts)
        return response.text
    except Exception as e:
        print(f"Error sending message to chatbot: {e}") # Log this error too!
        return f"Unable to process your request at this time. Due to the following reason: {str(e)}"
