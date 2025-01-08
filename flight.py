import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Part

#For training your AI, a good doc : https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env

# Tunned model 
# projects/pure-album-446616-n3/locations/us-central1/models/my_trained_model
# Model
# gemini-1.5-flash-002
model = GenerativeModel("projects/pure-album-446616-n3/locations/us-central1/models/397980328301428736") 

chat_session = None
if not project_id:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
    
vertexai.init(project=project_id,)

def getSession():
    global chat_session 
    if not chat_session:
        chat_session = model.start_chat() 
        system_prompt = """
        You are a consultant of a traveling agengcy, the client will ask you question about flights, destination and anything related to planning a trip. 
        Don't answer if the question is not related to traveling, or trip planning. Politely ask client to not to derail. 
        """
        response = chat_session.send_message(system_prompt)
    return chat_session

def getChatResponse(chat_session, prompt, links):
    message_parts = [prompt]
    for link in links:
        try:
            message_parts.append(Part.from_uri(link, "image/jpeg"))
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            return f"Could not process image link: {link}"  

    try:
        response = chat_session.send_message(message_parts)
        return response.text
    except Exception as e:
        print(f"Error sending message to chatbot: {e}") # Log this error too!
        return f"Unable to process your request at this time. Due to the following reason: {str(e)}"
