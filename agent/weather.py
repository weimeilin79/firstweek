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

def suggest_clothing(temp: float):
    return llm.invoke("What clothing do you recommend if the temperature is {temp} celsius)}")


def ask_llm(input_msg):
    prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant that help answering question and it should be all about travel advice "
                    )
                ),
                input_msg,
            ]
        )

    prompt = prompt_template.format()
    llm_with_tools = llm.bind_tools([get_temperature, suggest_clothing])
    return llm_with_tools.invoke(prompt)

def ask_llm_with_tool(state: MessagesState):
    print(f"---->{state["messages"]}")
    return {"messages": [ask_llm(state["messages"][0].content)]}



if __name__ == '__main__':
    mymsg =  "Get the current temperature of location lat 35.447246 and ln -85.069161, and suggest what cloth to wear"
    #print(mymsg)
    builder = StateGraph(MessagesState)
    builder.add_node("ask_llm_with_tool", ask_llm_with_tool)
    builder.add_node("tools", ToolNode([get_temperature]))
    builder.add_edge(START, "ask_llm_with_tool")
    builder.add_conditional_edges("ask_llm_with_tool",tools_condition)
    builder.add_edge("tools", "ask_llm_with_tool")
    graph = builder.compile()

    messages = graph.invoke({"messages": mymsg})
    print(messages)
    for m in messages['messages']:
        m.pretty_print()
    #print(get_temperature(35.447246,-85.069161))



