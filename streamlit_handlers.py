import streamlit as st
from streamlit_chat import message
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


def display_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_reset"] = False
        return
    if st.session_state["chat_reset"]:
        st.session_state["chat_reset"] = False

    chat_history = st.session_state["chat_history"]
    for index, tup in enumerate(chat_history):
        message(tup[0], is_user=True, key=f"msg_{index}_user")
        message(tup[1], is_user=False, key=f"msg_{index}_ai")


def submit():
    st.session_state["something"] = st.session_state["user_input"]
    st.session_state["user_input"] = ""
