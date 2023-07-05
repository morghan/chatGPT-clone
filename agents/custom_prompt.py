import openai
import os
import json
import requests
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from pprint import pprint
from agent_with_vector_store import agent2
from langchain.callbacks import get_openai_callback


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-0613"

custom_system_prompt = """   """


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, functions=None, function_call=None, model=GPT_MODEL
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(
                f"function ({message['name']}): {message['content']}\n"
            )
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[
                    messages[formatted_messages.index(formatted_message)]["role"]
                ],
            )
        )


def get_current_weather(location, format):
    return f"The current weather in {location} is 30 degrees {format}"


def respond_franchise_inquiry(inquiry):
    with get_openai_callback() as cb:
        response = agent2.run(inquiry)
        print(cb)
    return response


functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    },
    {
        "name": "respond_franchise_inquiry",
        "description": "Responds to inquiries about franchises. The inquiry should be a fully formed question.",
        "parameters": {
            "type": "object",
            "properties": {
                "inquiry": {
                    "type": "string",
                    "description": "The inquiry about a or multiple franchises.",
                },
            },
            "required": ["inquiry"],
        },
    },
]


def execute_function_call(message):
    if message["function_call"]["name"] == "get_current_weather":
        location = json.loads(message["function_call"]["arguments"])["location"]
        format = json.loads(message["function_call"]["arguments"])["format"]
        results = get_current_weather(location=location, format=format)
    elif message["function_call"]["name"] == "respond_franchise_inquiry":
        inquiry = json.loads(message["function_call"]["arguments"])["inquiry"]
        results = respond_franchise_inquiry(inquiry=inquiry)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results


messages = []
messages.append(
    {
        "role": "system",
        "content": """
        Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
        The user can make questions about the following franchises: Cookie Cutters, Five Star Bath Solutions.
        """,
    }
)
# messages.append({"role": "user", "content": "What's the weather like today"})
# chat_response = chat_completion_request(messages, functions=functions)
# assistant_message = chat_response.json()["choices"][0]["message"]
# messages.append(assistant_message)
# pretty_print_conversation(messages)
while True:
    query = input("💬 To exit type 'q', else Enter a query for GPT: ")
    if query == "q":
        break
    print("\n\n---------------------------\n")
    messages.append({"role": "user", "content": query})
    chat_response = chat_completion_request(messages, functions=functions)
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    if assistant_message.get("function_call"):
        results = execute_function_call(assistant_message)
        messages.append(
            {
                "role": "function",
                "name": assistant_message["function_call"]["name"],
                "content": results,
            }
        )
    pretty_print_conversation(messages)
    pprint(chat_response.json()["usage"])
    print("\n---------------------------\n\n")
