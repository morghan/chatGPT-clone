from dotenv import load_dotenv
import os
import psycopg2

import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.callbacks import get_openai_callback

st.set_page_config(page_title="ChatGPT Clone", page_icon="💬")


@st.cache_data
def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY not found")
        exit(1)
    else:
        st.header("ChatGPT-driven assessment 📄")
        st.subheader(
            " ⬅️ You can freely chat with the AI assistant by typing in the box to the left . You can also upload (txt file) a system prompt for the assistant."
        )
        print("OPENAI_API_KEY found")


@st.cache_resource
def connect_db():
    return psycopg2.connect(**st.secrets["postgres"])


db_connection = connect_db()


def fetch_system_prompt():
    print("fetching system prompt from db")
    with db_connection.cursor() as cur:
        query = "SELECT * FROM system_prompt;"
        try:
            cur.execute(query)
            system_prompt = cur.fetchall()
            return system_prompt
        except Exception as e:
            print("Error=>", e)
            return None


def display_chat_history():
    if "chat_history" not in st.session_state:
        return
    chat_history = st.session_state["chat_history"]
    for index, msg in enumerate(chat_history[1:]):
        if index % 2 == 0:
            message(msg.content, is_user=True, key=f"msg_{index}_user")
        else:
            message(msg.content, is_user=False, key=f"msg_{index}_ai")


def upload_file(file_content):
    print("uploading file to db")
    with db_connection.cursor() as cur2:
        sql = "SELECT * FROM insert_update_prompt(%s);"
        try:
            cur2.execute(sql, (file_content,))
            db_connection.commit()
        except (Exception, psycopg2.DatabaseError) as e:
            print("Error:", e)
            db_connection.rollback()


def main():
    init()
    chat = ChatOpenAI(temperature=0.2)
    system_prompt = fetch_system_prompt()
    # st.write("system_prompt", system_prompt)
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload your system prompt from .txt file", type=["txt"]
        )
        if uploaded_file is not None:
            st.write("file uploaded successfully✅")
            st.write("Here is the content of your file:")
            file_content = uploaded_file.getvalue().decode("utf-8")
            upload_file(file_content)
            system_prompt = fetch_system_prompt()
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]

        print("system_prompt", system_prompt)
        # st.write("system_prompt", system_prompt)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = (
                [SystemMessage(content=f"{system_prompt[0][2]}")]
                if system_prompt is not None
                else [
                    SystemMessage(
                        content=f"""You are a friendly AI assistant that interacts with the user.
          The user may have questions about various topics.
          If you don't know the answer,respond honestly that you don't know."""
                    ),
                ]
            )
        # st.write("session_state", st.session_state)
        user_input = st.text_input("Your input", key="user_input")

    if user_input:
        st.session_state["chat_history"].append(HumanMessage(content=user_input))

        with get_openai_callback() as callback, st.spinner("Thinking..."):
            ai_response = chat(st.session_state["chat_history"])
            print(callback)

        st.session_state["chat_history"].append(AIMessage(content=ai_response.content))

    display_chat_history()


if __name__ == "__main__":
    main()
