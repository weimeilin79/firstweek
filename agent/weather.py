#from langchain_google_vertexai import VertexAI
import os
import random
import requests
import vertexai
from typing import TypedDict, Literal
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition



# ENV SETUP
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env

# Connect to resourse needed from Google Cloud
llm = ChatVertexAI(model_name="gemini-1.5-flash-002")

def get_temperature(lat: float, lng: float):   
    """Get the current temperature of a given location

    Args:
        lat: Latitude float
        lng: Longitude float
    """
    print(f"lat: {lat}, lng: {lng}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=temperature_2m"
    response = requests.get(url)
    data = response.json()
    temperature = data["current"]["temperature_2m"]
    return temperature

def suggest_clothing(temperature: float):
    """Suggest and recommend to wear base on temperature given


    Args:
        temperature: temperature float
    """
    query = f"What clothing do you recommend if the temperature is {temperature} celsius?"
    print(f"-------->{query}")
    response = llm.invoke(query)
    print(f"CLOTHING RESPONSE------------>temperature: {response.content}")
    return {"messages": response.content}
    
def find_location_name(lat: float, lng: float):
    """Base on the given latitude and longitude, provide the name like town (if applicable), state(if applicable) and country(if applicable) of the location
    
    Args:
        lat: Latitude float
        lng: Longitude float
    """
    query = f"What is the name like town (if applicable), state(if applicable) and country(if applicable) of the location in lat {lat} and lng {lng}?"
    print(f"-------->{query}")
    response = llm.invoke(query)
    print(f"Location Name RESPONSE------------>temperature: {response.content}")
    return {"messages": response.content}

tools = [get_temperature, suggest_clothing,find_location_name]

def ask_llm(state: MessagesState):

    sys_msg = SystemMessage(
                    content=(
                        "You are a helpful assistant that help answering question on weather and travel advice "
                    )
                )

    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])} 




def find_location_info(lat: float, lng: float):
    mymsg =  f"What's the current temperature of location lat {lat} and ln {lng}? What's the name of the location? Suggest what to wear base on the temperature"
    #print(mymsg)
    builder = StateGraph(MessagesState)
    builder.add_node("ask_llm_with_tool", ask_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "ask_llm_with_tool")
    builder.add_conditional_edges("ask_llm_with_tool",tools_condition)
    builder.add_edge("tools", "ask_llm_with_tool")
    graph = builder.compile()

    messages = graph.invoke({"messages": mymsg})
    print(messages)
    for m in messages['messages']:
        m.pretty_print()
    #print the last item in  messages['messages']
    return messages['messages'][-1].content





