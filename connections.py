import psycopg2
import pinecone
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# This must be the first streamlit command called, otherwise it won't work
st.set_page_config(page_title="ChatGPT Clone", page_icon="💬")

# Connect to pinecone kb
pinecone.init(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["env"],
)


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
            cur2.execute(sql, (file_content.strip(),))
            db_connection.commit()
        except Exception as e:
            print("Error:", e)
            db_connection.rollback()


def fetch_namespaces():
    # List all indexes in your Pinecone account
    active_indexes = pinecone.list_indexes()

    # Get namespaces for index called pdf-kb which is the only index at the moment
    pinecone_index = pinecone.Index(active_indexes[0])
    index_description = pinecone_index.describe_index_stats()
    namespaces = index_description.get("namespaces")

    return list(namespaces)
