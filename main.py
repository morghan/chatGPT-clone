import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os, openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.callbacks import get_openai_callback


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY not found")
        exit(1)
    else:
        print("OPENAI_API_KEY found")


def main():
    init()

    chat = ChatOpenAI(temperature=0.2)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            SystemMessage(
                content=f"""You are a friendly AI assistant that interacts with the user.
            The user may have questions about various topics. 
            If you don't know the answer,respond honestly that you don't know."""
            ),
        ]

    st.set_page_config(page_title="ChatGPT Clone", page_icon="💬")
    st.header("ChatGPT-driven assessment 📄")

    with st.sidebar:
        user_input = st.text_input("Your input", key="user_input")

    if user_input:
        st.session_state["chat_history"].append(HumanMessage(content=user_input))

        with get_openai_callback() as callback, st.spinner("Thinking..."):
            ai_response = chat(st.session_state["chat_history"])
            print(callback)

        st.session_state["chat_history"].append(AIMessage(content=ai_response.content))

    # Display chat history
    chat_history = st.session_state["chat_history"]
    for index, msg in enumerate(chat_history[1:]):
        if index % 2 == 0:
            message(msg.content, is_user=True, key=f"msg_{index}_user")
        else:
            message(msg.content, is_user=False, key=f"msg_{index}_ai")


if __name__ == "__main__":
    main()
