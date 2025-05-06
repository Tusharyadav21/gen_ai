import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/"
)

query = input("> ")

result = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "user", "content": query},
    ],
)

print(result.choices[0].message.content)
