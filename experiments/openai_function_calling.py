import os
from dotenv import load_dotenv
import openai

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY not found")
    exit(1)
else:
    print("OPENAI_API_KEY found")
    openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=[{"role": "user", "content": "Hello there"}],
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ],
)
print(completion)
