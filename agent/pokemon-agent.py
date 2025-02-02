import vertexai
import os

from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)


project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
# Initialize Vertex AI
vertexai.init(project=project_id, location="us-central1")

# Initialize Gemini model
model = GenerativeModel("gemini-1.5-flash-002")

user_prompt_content = Content(
    role="user",
    parts=[
        Part.from_text("Tell me about Pikachu?"),
    ],
)


get_pokemon_ability_func = FunctionDeclaration(
    name="get_pokemon_ability",
    description="Get the abilities of a given pokemon",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Pokemon name"}}
    },  
)

get_pokemon_stats_func = FunctionDeclaration(
    name="get_pokemon_stats",
    description="Get the stats of a given pokemon",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Pokemon name"}}
    },  
)

def get_pokemon_ability(pokemon):
    #generate random abilities of pikachu return as a list of abilities in json format
    abilities = ["Static", "Lightning Rod", "Z-Moves", "Thunderbolts"]
    return {"abilities": abilities}

def get_pokemon_stats(pokemon):
    #generate random stats of pikachu return json format
    stats = {
        "hp": 35,
        "attack": 55,
        "defense": 40,
        "special-attack": 50,
        "special-defense": 50,
        "speed": 90
    }
    return stats
    
# Define a tool that includes the above get_current_weather_func
pokemon_tool = Tool(
    function_declarations=[get_pokemon_ability_func,get_pokemon_stats_func],
)

# Send the prompt and instruct the model to generate content using the Tool that you just created
response = model.generate_content(
    user_prompt_content,
    generation_config=GenerationConfig(temperature=0),
    tools=[pokemon_tool],
)
#function_call = response.candidates[0].function_calls[0]
#print(response)

# Loop through the response.candidates and print it out
"""
for candidate in response.candidates:
    print(f"----->{candidate.content}")
    for function_call in candidate.function_calls:
        print(f"----->{function_call}")
"""
json_schema_1= {
  "type": "object",
  "properties": {
    "abilities": {
      "type": "array",
      "items": []
    }
    
  }
}
#print(f"response.candidates[0].content----->{response.candidates[0].content}")
api_resonses=[]
for candidate in response.candidates:
    print(f"----->{candidate.content}")
    for function_call in candidate.function_calls:
      calling_function_string = f"{function_call.name}({function_call.args})"
      print(f"===>{calling_function_string}")
      response_api = eval(calling_function_string)
      api_resonses.append(Part.from_function_response(name=function_call.name, response=response_api))

print(api_resonses)

response = model.generate_content(
  [
    user_prompt_content,
    response.candidates[0].content,  
    Content( parts=api_resonses),
  ],
  tools=[pokemon_tool],
)

print(response.text)




#Loop through 


def get_temperature(lat, lng):   
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=temperature_2m"
    response = requests.get(url)
    data = response.json()
    temperature = data["current"]["temperature_2m"]
    return temperature

