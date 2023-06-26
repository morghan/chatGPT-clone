import psycopg2
import pinecone
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# This must be the first streamlit command called, otherwise it won't work
st.set_page_config(page_title="ChatGPT Clone", page_icon="💬")


# These are functions streamlit runs when it starts
# But then not run again because it is cached
@st.cache_resource
def connect_db():
    try:
        print("connecting to db...")
        return psycopg2.connect(**st.secrets["postgres"])
    except Exception as e:
        print("Error=>", e)
        return None


@st.cache_resource
def load_vector_store():
    try:
        print("loading vector store...")
        embeddings = OpenAIEmbeddings()
        pinecone.init(
            api_key=st.secrets["pinecone"]["api_key"],
            environment=st.secrets["pinecone"]["env"],
        )
        vectors = Pinecone.from_existing_index(
            index_name=st.secrets["pinecone"]["index_name"],
            embedding=embeddings,
            namespace=st.secrets["pinecone"]["namespace"],
        )
        return vectors
    except Exception as e:
        print("Error=>", e)
        return None


db_connection = connect_db()
vector_store = load_vector_store()


def fetch_system_prompt():
    print("fetching system prompt from db")
    with db_connection.cursor() as cur:
        query = "SELECT * FROM system_prompt;"
        try:
            cur.execute(query)
            return cur.fetchall()
        except Exception as e:
            print("Error=>", e)
            return None


def upload_prompt(file_content):
    print("uploading file to db")
    with db_connection.cursor() as cur2:
        sql = "SELECT * FROM insert_update_prompt(%s);"
        try:
            cur2.execute(sql, (file_content,))
            db_connection.commit()
        except Exception as e:
            print("Error:", e)
            db_connection.rollback()
