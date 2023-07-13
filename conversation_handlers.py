import streamlit as st
import openai
import os
import json
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.callbacks import get_openai_callback


openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-16k-0613"


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, functions=None, function_call=None, model=GPT_MODEL
):
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer " + openai.api_key,
    # }
    # json_data = {"model": model, "messages": messages, "temperature": 0.3}
    # if functions is not None:
    #     json_data.update({"functions": functions})
    # if function_call is not None:
    #     json_data.update({"function_call": function_call})
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions if functions is not None else [],
            function_call=function_call if function_call is not None else "auto",
            stream=True,
        )
        # response = requests.post(
        #     "https://api.openai.com/v1/chat/completions",
        #     headers=headers,
        #     json=json_data,
        # )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def respond_franchise_inquiry(inquiry, st_callback=None):
    if st.session_state.get("agent") is not None:
        agent = st.session_state["agent"]
        return agent.run(inquiry, callbacks=[st_callback])
    else:
        return "I'm sorry, I don't have an QA Agent to respond to your inquiry."


def execute_function_call(message, st_callback=None):
    if message["function_call"]["name"] == "respond_franchise_inquiry":
        inquiry = json.loads(message["function_call"]["arguments"])["inquiry"]
        results = respond_franchise_inquiry(inquiry=inquiry, st_callback=st_callback)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results
