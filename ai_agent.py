import os
import json
import requests
from openai import OpenAI

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text.strip()}"

    return "Something went wrong"


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather of that city.",
    },
}

system_prompt = """
You are a helpful AI assistant specialized in resolving user queries using tools.
You follow a loop of steps: start, plan, action, observe, and output.

For a given query:
1. Plan how to solve it step by step.
2. Select the appropriate tool from the available tools.
3. Perform the action by calling the tool.
4. Observe the tool's output.
5. Use the observation to resolve the query.

Rules:
- Follow the output JSON format.
- Perform one step at a time and wait for the next input.
- Carefully analyze the query.

Output JSON Format:
{
    "step": "string",
    "content": "string",
    "function": "string or null",
    "input": "string or null"
}

Available Tools:
- get_weather: Takes a city name as input and returns the current weather.

Example:
User Query: What is the weather of New York?
Step-by-step:
{"step": "plan", "content": "The user wants weather data for New York."}
{"step": "plan", "content": "Use get_weather to retrieve the weather."}
{"step": "action", "function": "get_weather", "input": "New York"}
{"step": "observe", "content": "12°C and sunny."}
{"step": "output", "content": "The weather in New York is 12°C and sunny."}
"""

messages = [{"role": "system", "content": system_prompt}]

while True:
    user_query = input("> ")
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_output = response.choices[0].message.content
        parsed_output = json.loads(parsed_output)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        step = parsed_output["step"]

        if step == "plan":
            print(f"🧠 {parsed_output['content']}")
            continue

        if step == "action":
            tool_name = parsed_output["function"]
            tool_input = parsed_output["input"]

            if tool_name in available_tools:
                tool_fn = available_tools[tool_name]["fn"]
                result = tool_fn(tool_input)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "content": result}),
                    }
                )
            continue

        if step == "output":
            print(f"🤖 {parsed_output['content']}")
            break
