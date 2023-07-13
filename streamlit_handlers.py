import streamlit as st
from dotenv import load_dotenv
import os


@st.cache_data(ttl=60 * 60)
def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY not found")
        exit(1)
    else:
        print("OPENAI_API_KEY found")
        st.header("👨‍💻QUALIFYI - AI Broker")
        st.subheader(
            "You can freely chat with the AI Broker. You can also upload (txt file) a system prompt for the broker to follow."
        )


def render_conversation():
    for message in st.session_state["chat_history"][1:]:
        if message["content"] is not None:
            with st.chat_message(
                name=message["role"],
                avatar="https://raw.githubusercontent.com/morghan/chatGPT-clone/main/icons/database.png" if message["role"] == "function" else None,
            ):
                st.markdown(message["content"])


def render_qa_agent():
    if st.session_state.get("agent") is not None:
        with st.expander("📚 QA Agent"):
            for index, tool in enumerate(st.session_state["agent"].tools):
                st.write(f"**Tool name**: {tool.name}")
                st.write(f"**Tool description**: {tool.description}")
                if index != len(st.session_state["agent"].tools) - 1:
                    st.divider()
