import json
import os
from openai import OpenAI

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

system_prompt = """
You are an AI assistant that breaks down complex problems into clear steps before answering.

For each user query:
1. “analyse” the question.
2. “think” through at least 5–6 sub-steps.
3. Produce an “output” (your raw answer).
4. “validate” your output.
5. Provide the final “result.”

Rules:
- Emit exactly one JSON object per step, using:
  { "step": "...", "content": "..." }
- Do one step at a time and wait for the next loop.
- Carefully analyse the user query.

Example:
User: What is 2 + 2  
→ { "step":"analyse",  "content":"Identify that this is a simple arithmetic addition." }  
→ { "step":"think",    "content":"Plan to add 2 and 2 to get the result." }  
→ { "step":"output",   "content":"4" }  
→ { "step":"validate", "content":"4 is correct for 2 + 2." }  
→ { "step":"result",   "content":"2 + 2 = 4." }
"""

messages = [
    {"role": "system", "content": system_prompt},
]

query = input("> ")
messages.append({"role": "user", "content": query})

while True:
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        messages=messages,
    )
    parsed = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed)})

    step = parsed.get("step")
    content = parsed.get("content")

    # Print each thought until 'result'
    if step != "result":
        print(f"🧠 [{step}]: {content}")
        continue

    print(f"🤖: {content}")
    break
