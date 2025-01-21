import os
import sys
import requests
import json
from datetime import date
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd

import osmnx as ox
import networkx as nx

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool

from typing import Type
from pydantic import BaseModel, Field


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Raio da Terra em quilômetros

    return c * r

def find_nearest_risk_point(lat, lon, data):
    distances = data.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)

    nearest_index = distances.idxmin()
    
    return data.loc[nearest_index, 'RL']

def fetch_hbi_data():
    def endpoint(page):
        return f"https://fixmyberlin.de/api/v1/sections?page={page}&page_size=200"
    
    page = 1
    has_next = True
    data = []
    
    while has_next:
        try:
            resp = requests.get(endpoint(page)).json()
            data += resp["results"]
            has_next = resp["next"] is not None
            page += 1
        except Exception as e:
            print(f"Error fetching HBI data: {e}")
            break
    
    # Process HBI data
    by_borough = defaultdict(lambda: {
        "count": 0,
        "total_length": 0.0,
        "hbi_avg_product": 0.0
    })
    
    for entry in data:
        target = by_borough[entry["borough"]]
        target["count"] += 1
        for side in entry["details"]:
            target["total_length"] += side["length"]
            target["hbi_avg_product"] += side["length"] * side["happy_bike_index"]
    
    # Calculate average HBI for each borough
    hbi_data = {}
    borough_hbi_values = []  # Store only the HBI values
    for name, borough in by_borough.items():
        hbi_avg = borough["hbi_avg_product"] / borough["total_length"] if borough["total_length"] > 0 else 0
        hbi_data[name] = hbi_avg
        borough_hbi_values.append(hbi_avg)
    
    # Add HBI summary stats as a separate field
    if borough_hbi_values:
        hbi_data['summary'] = {
            'average': sum(borough_hbi_values) / len(borough_hbi_values),
            'min': min(borough_hbi_values),
            'max': max(borough_hbi_values)
        }
    
    return hbi_data


class RiskPointInput(BaseModel):
    lat: float = Field(..., title="Latitude", description="Latitude of the point")
    lon: float = Field(..., title="Longitude", description="Longitude of the point")

class RiskTripInput(BaseModel):
    start_lat: float = Field(..., title="Latitude of the starting point", description="Latitude of the starting point")
    start_lon: float = Field(..., title="Longitude of the starting point", description="Longitude of the starting point")
    end_lat: float = Field(..., title="Latitude of the ending point", description="Latitude of the ending point")
    end_lon: float = Field(..., title="Longitude of the ending point", description="Longitude of the ending point")

class RiskPointTool(BaseTool):
    name = "RiskPointTool"
    description = "A tool that computes the risk of a point in the map"
    args_schema: Type[BaseModel] = RiskPointInput

    def _run(
        self,
        lat: float,
        lon: float
    ) -> str:
        data = pd.read_csv('risk_map.csv', sep=";")
        risk_lane = find_nearest_risk_point(lat, lon, data)

        return risk_lane

class RiskTripTool(BaseTool):
    name = "RiskTripTool"
    description = "A tool that computes the risk of a trip in the map"
    args_schema: Type[BaseModel] = RiskTripInput

    def _run(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float
    ) -> str:
        data = pd.read_csv('risk_map.csv', sep=";")
        place_name = "Porto, Portugal"
        graph = ox.graph_from_place(place_name, network_type='all')

        # Fetch HBI data and get average (excluding summary)
        hbi_data = fetch_hbi_data()
        avg_hbi = hbi_data['summary']['average'] if hbi_data and 'summary' in hbi_data else 0

        start = (start_lat, start_lon)
        end = (end_lat, end_lon)

        start_node = ox.distance.nearest_nodes(graph, start[1], start[0])
        end_node = ox.distance.nearest_nodes(graph, end[1], end[0])

        route = nx.shortest_path(graph, start_node, end_node, weight='length')

        risk = {
            1: 0,
            2: 0,
            3: 0
        }

        for node in route:
            node_data = graph.nodes[node]
            base_risk = find_nearest_risk_point(node_data['y'], node_data['x'], data)
            
            # Adjust risk based on HBI
            # If HBI is high (> 3), decrease risk level by 1
            # If HBI is low (< 2), increase risk level by 1
            if avg_hbi > 3 and base_risk > 1:
                adjusted_risk = base_risk - 1
            elif avg_hbi < 2 and base_risk < 3:
                adjusted_risk = base_risk + 1
            else:
                adjusted_risk = base_risk
            
            risk[adjusted_risk] += 1

        sum_risk = risk[1] + risk[2] + risk[3]

        # Format HBI information
        hbi_info = ""
        if hbi_data:
            hbi_info = (f"\nHBI Statistics:"
                       f"\n- Average HBI: {avg_hbi:.2f}"
                       f"\n- Impact: {'Reducing' if avg_hbi > 3 else 'Increasing' if avg_hbi < 2 else 'Neutral'} risk levels")

        return (f"Low risk percentage: {risk[1]/sum_risk*100:.2f}%, "
                f"Medium risk percentage: {risk[2]/sum_risk*100:.2f}%, "
                f"High risk percentage: {risk[3]/sum_risk*100:.2f}% "
                f"(HBI adjusted){hbi_info}")

openai_api_key = os.getenv("OPENAI_API_KEY")

chat_open_ai = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

risk_point_tool = RiskPointTool()
risk_trip_tool = RiskTripTool()

tools = [
    risk_point_tool,
    risk_trip_tool
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an agent responsible for assisting users in computing cycling routes based on risk levels and Happy Bike Index (HBI). Your task is to analyze the computed risk levels, HBI data, and provide comprehensive and well-tailored responses to user inquiries.

            Risk Levels:
            - (1) Low Risk: Routes that are safe for most cyclists, including beginners. These routes are well within the city's emergency mitigation capacity, with close proximity to emergency services.
            - (2) Medium Risk: Routes that may have moderate distance from emergency services. These routes are suitable for experienced cyclists who are comfortable with occasional risks and may require emergency services less readily available.
            - (3) High Risk: Routes that are far from emergency services and other emergency-related amenities. These routes involve significant hazards and are recommended only for very experienced cyclists or those seeking a challenge.

            Happy Bike Index (HBI):
            - The HBI is a measure of cycling infrastructure quality and safety
            - HBI ranges from 1 (poor) to 5 (excellent)
            - High HBI (> 3) reduces the risk level of a route
            - Low HBI (< 2) increases the risk level of a route
            - HBI factors include:
              * Quality of cycling infrastructure
              * Separation from traffic
              * Surface quality
              * Overall cycling comfort

            Main Instructions:
            1. Input Interpretation:
            - Consider both the computed risk level (1, 2, or 3) and HBI data
            - Understand how HBI affects the final risk assessment
            - Factor in emergency service proximity and infrastructure quality

            2. Response Generation:
            - Provide clear descriptions of the route based on both risk level and HBI
            - Explain how HBI has influenced the risk assessment
            - Include relevant safety tips specific to the infrastructure quality
            - Suggest safer alternatives if the risk is high or HBI is low

            3. Example Responses:
            - Good Infrastructure: "This route has a medium base risk level, but thanks to high-quality cycling infrastructure (HBI: 4.2), the effective risk is lower. You'll find dedicated bike lanes and good separation from traffic."
            - Poor Infrastructure: "While emergency services are nearby, the poor cycling infrastructure (HBI: 1.8) increases the effective risk. Consider alternative routes with better bike lanes."

            4. General Tips:
            - Always remind users to wear appropriate safety gear
            - Encourage weather condition checks
            - Advise carrying repair kits and staying hydrated
            - Suggest carrying a mobile phone and sharing route details

            Your goal is to enhance cycling experiences while prioritizing safety, considering both emergency service proximity and infrastructure quality through HBI data.
            """
        ),
        (
            "user",
            "{input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = chat_open_ai.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

user_input = sys.argv[sys.argv.index("--prompt") + 1]

result = list(
    agent_executor.stream(
        {"input": user_input}
    )
)

print(result[-1]["messages"][0].content)
