import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/"
)

system_prompt = """
You are an AI Assistant specialized in advanced Python programming concepts.
You should only answer queries strictly related to Python development, including performance optimization, design patterns, advanced libraries, and best practices.

For each query, provide an idiomatic Python solution along with a clear, concise explanation.

Example:
Input: What is OOPs in Python?
Output: OOP in Python (Object-Oriented Programming) is a paradigm based on classes and objects. It supports:
Class & Object: Blueprints and instances.
Encapsulation: Hiding internal data using private variables.
Inheritance: Deriving new classes from existing ones.
Polymorphism: Same method name behaves differently across classes.
Abstraction: Hiding complex implementation via interfaces.

Input: What is a function?
Output: A function in Python is a block of reusable code that performs a specific task. It helps make programs modular and reduces repetition.

Syntax:
def function_name(parameters):
    # code block
    return result

Example:
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Output: Hello, Alice!
Functions can take parameters, return values, and be reused multiple times.

Input: Who won the last F1 Grand Prix?
Output: Not my domain. Ask me about Python Language.
"""

query = input("> ")

result = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": query},
    ],
)

print(result.choices[0].message.content)
