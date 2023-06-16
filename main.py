from dotenv import load_dotenv
import os
import psycopg2
import pinecone

from PyPDF2 import PdfReader
import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, Redis
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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


# This is the first function streamlit runs when it starts
# But then it is not run again because it is cached
@st.cache_resource
def connect_db():
    print("connecting to db")
    return psycopg2.connect(**st.secrets["postgres"])


@st.cache_resource
def load_vector_store():
    print("loading vector store")
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


def display_chat_history():
    if "chat_history" not in st.session_state:
        return
    chat_history = st.session_state["chat_history"]
    # for index, msg in enumerate(chat_history[1:]):
    #     if index % 2 == 0:
    #         message(msg.content, is_user=True, key=f"msg_{index}_user")
    #     else:
    #         message(msg.content, is_user=False, key=f"msg_{index}_ai")
    for index, tup in enumerate(chat_history):
        message(tup[0], is_user=True, key=f"msg_{index}_user")
        message(tup[1], is_user=False, key=f"msg_{index}_ai")


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


def create_qa_chain(template_prompt, chat_model, memory, vector_store):
    if "context" in template_prompt[0][2] and "question" in template_prompt[0][2]:
        system_prompt = PromptTemplate.from_template(template=template_prompt[0][2])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": system_prompt},
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            chain_type="stuff",
            verbose=True,
        )
    else:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            memory=memory,
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            chain_type="stuff",
            verbose=True,
        )
    return qa_chain


def main():
    init()
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.2)
    # qa_chain = load_qa_chain(llm=chat_model, chain_type="stuff")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    template = fetch_system_prompt()

    qa_chain = create_qa_chain(template, chat_model, memory, vector_store)

    with st.sidebar:
        # System Message upload
        uploaded_file = st.file_uploader(
            "You can customize the system prompt by uploading a .txt file. This prompt will be saved for future sessions",
            type=["txt"],
        )
        if uploaded_file is not None:
            st.write("file uploaded successfully✅")
            st.write("Here is the content of your file:")
            file_content = uploaded_file.getvalue().decode("utf-8")
            upload_file(file_content)
            template = fetch_system_prompt()
            qa_chain = create_qa_chain(template, chat_model, memory, vector_store)
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]

        # PDF file upload and storage in vector store
        pdf = st.file_uploader(
            "Upload your PDF. It may take some time, but it will be saved for future sessions",
            type=["pdf"],
        )
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            st.write("PDF file uploaded successfully✅")

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            st.write("chunks length", len(chunks))
            if vector_store is not None:
                with st.spinner("adding text to vector store"):
                    try:
                        ids = vector_store.add_texts(texts=chunks, namespace="fdd")
                        st.write("text added successfully✅", "ids:", ids)
                    except Exception as e:
                        st.write("Error:", e)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Your input", key="user_input")

    if user_input:
        with get_openai_callback() as callback, st.spinner("Thinking..."):
            ai_response = qa_chain(
                {
                    "question": user_input,
                }
            )
            print(callback)

        # st.session_state["chat_history"].append(AIMessage(content=ai_response.content))
        st.session_state["chat_history"].append((user_input, ai_response["answer"]))

        display_chat_history()


if __name__ == "__main__":
    main()
