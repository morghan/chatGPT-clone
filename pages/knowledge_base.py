import streamlit as st
import time
from streamlit_tree_select import tree_select
from streamlit_modal import Modal
import streamlit.components.v1 as components
from connections import fetch_namespaces, delete_namespaces
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


def build_directory():
    st.session_state["namespaces"] = fetch_namespaces()
    st.session_state["directory"] = [
        {"label": namespace, "value": namespace}
        for namespace in st.session_state["namespaces"]
    ]


def delete_resources():
    result = delete_namespaces(st.session_state["directory_data"]["checked"])
    if result:
        build_directory()
    return result


st.title("🗄️ Remote knowledge base")
if "namespaces" not in st.session_state and "directory" not in st.session_state:
    build_directory()


if "directory_data" not in st.session_state:
    st.session_state["directory_data"] = {}

if "namespaces_deleted" not in st.session_state:
    st.session_state["namespaces_deleted"] = False

st.subheader("Add resources")
with st.form("pdf_upload2", clear_on_submit=True):
    franchise_name = st.text_input(
        "Franchise name", help="No numbers or special characters"
    )
    pdf = st.file_uploader(
        "Upload your PDF. It may take some time, but it will be saved for future sessions",
        type=["pdf"],
    )

    submitted = st.form_submit_button("Submit")

    if submitted and franchise_name is not None:
        if (
            all(part.isalpha() for part in franchise_name.split(sep=" "))
            and franchise_name not in st.session_state["namespaces"]
        ):
            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                chunks = []
                with st.spinner("Preparing content 🚧 ..."):
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    if text != "":
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200, length_function=len
                        )
                        chunks = text_splitter.split_text(text=text)
                        st.success(f"✅ Content ready!")

                    else:
                        st.error("Invalid PDF file. PDF file does not have text.")
                with st.spinner("Uploading content 🚧 ..."):
                    embeddings = OpenAIEmbeddings()
                    vectors = Pinecone.from_texts(
                        texts=chunks,
                        embedding=embeddings,
                        batch_size=80,
                        index_name=st.secrets["pinecone"]["index_name"],
                        namespace=franchise_name,
                    )
                    build_directory()
                    st.success(f"✅ File uploaded successfully!")
            else:
                st.error("No PDF file selected. Please upload one.")
        else:
            st.error(
                f"Invalid name: **{franchise_name}**. Franchise Name already taken or has numbers and/or special characters."
            )

st.subheader("Select your resources")
st.session_state["directory_data"] = tree_select(st.session_state["directory"])

st.write(st.session_state["directory_data"])

modal = Modal("Confirm to Delete Resources", key="ModKey")

if len(st.session_state["directory_data"]["checked"]) > 0:
    disable_modal = False

else:
    disable_modal = True

open_modal = st.button("Delete Resources", disabled=disable_modal)
if open_modal:
    modal.open()

resources_list = st.session_state["directory_data"]["checked"]
string_list = ", ".join(resources_list)


if modal.is_open():
    with modal.container():
        st.write("*Are you sure you wish to delete the resources selected?*")
        st.write(string_list)

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if st.button("Yes"):
                if delete_resources() == True:
                    st.success("Resources successfully deleted!")
                    time.sleep(2)
                    modal.close()
                else:
                    st.error("Error: Unable to delete resources.")

        with col2:
            if st.button("No"):
                modal.close()
