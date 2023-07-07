import streamlit as st
import time
from streamlit_tree_select import tree_select
from streamlit_modal import Modal
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from main import build_custom_prompt_suffix
from connections import fetch_namespaces, delete_namespaces
from langchain_handlers import create_qa_agent


def build_directory():
    st.session_state["namespaces"] = fetch_namespaces()
    st.session_state["directory"] = [
        {"label": namespace, "value": namespace}
        for namespace in st.session_state["namespaces"]
    ]


def delete_resources(namespaces_to_delete):
    result = delete_namespaces(namespaces_to_delete)

    if result:
        # Python List Comprehension - Remove namespaces to delete from existing QA Agent
        updated_agent_namespaces = [
            x
            for x in st.session_state["agent_namespaces"]
            if x not in set(namespaces_to_delete)
        ]
        if len(updated_agent_namespaces) > 0:
            is_agent_updated = create_agent(updated_agent_namespaces)
            if is_agent_updated:
                return result
        build_directory()
    return result


def check_directory():
    if len(st.session_state["directory_data"]["checked"]) > 0:
        selected_resources = ", ".join(st.session_state["directory_data"]["checked"])
        return False, selected_resources
    else:
        return True, None


def create_agent(namespaces):
    st.session_state["agent_namespaces"] = namespaces
    try:
        st.session_state["agent"] = create_qa_agent(
            st.session_state["agent_namespaces"]
        )
        build_directory()
        build_custom_prompt_suffix()
        return True
    except Exception as e:
        print(e)
        return False


st.title("🗄️ Remote knowledge base")
if "namespaces" not in st.session_state and "directory" not in st.session_state:
    build_directory()

if "directory_data" not in st.session_state:
    st.session_state["directory_data"] = {}

if "agent_namespaces" not in st.session_state:
    st.session_state["agent_namespaces"] = []

if "agent" not in st.session_state:
    st.session_state["agent"] = None

if (
    st.session_state["agent"] is not None
    and "The user can make questions about the following franchises: "
    not in st.session_state["functions_instructions"]
):
    st.session_state[
        "functions_instructions"
    ] += f"The user can make questions about the following franchises: {', '.join(st.session_state['agent_namespaces'])}\n"
    st.session_state["messages"][0] = {
        "role": "system",
        "content": st.session_state["functions_instructions"]
        + st.session_state["custom_prompt"],
    }

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
                    # vectors = vector_store.add_texts(
                    #     texts=chunks,
                    #     namespace=franchise_name,
                    #     batch_size=80,
                    # )
                    embeddings = OpenAIEmbeddings()
                    temp_vector_store = Pinecone.from_texts(
                        texts=chunks,
                        embedding=embeddings,
                        batch_size=128,
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

delete_namespaces_col, create_agent_col = st.columns([0.3, 0.7])

delete_namespaces_modal = Modal("Confirm to Delete Resources", key="delete_namespaces")
create_agent_modal = Modal("Confirm Resources to create QA Agent", key="create_agent")

disable_modal, selected_resources = check_directory()

with delete_namespaces_col:
    open_delete_namespaces_modal = st.button("Delete Resources", disabled=disable_modal)
with create_agent_col:
    open_create_agent_modal = st.button("Create QA Agent", disabled=disable_modal)

if open_delete_namespaces_modal:
    delete_namespaces_modal.open()

if delete_namespaces_modal.is_open():
    with delete_namespaces_modal.container():
        st.write("**Are you sure you wish to delete the selected resources?**")
        st.write(
            "*Note: If the resource is part of the QA Agent,this will also remove it from the Agent.*"
        )
        st.write(selected_resources)

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if st.button("Yes"):
                if (
                    delete_resources(st.session_state["directory_data"]["checked"])
                    == True
                ):
                    st.success("Resources deleted and QA Agent updated successfully!")
                    time.sleep(2)
                    delete_namespaces_modal.close()
                else:
                    st.error("Error: Unable to delete resources.")

        with col2:
            if st.button("No"):
                delete_namespaces_modal.close()

if open_create_agent_modal:
    create_agent_modal.open()

if create_agent_modal.is_open():
    with create_agent_modal.container():
        st.write(
            "*Are you sure you wish to create the QA Agent with the resources selected?*"
        )
        st.write(selected_resources)

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if st.button("Yes"):
                if create_agent(st.session_state["directory_data"]["checked"]) == True:
                    st.success("QA Agent created!")
                    time.sleep(2)
                    create_agent_modal.close()
                else:
                    st.error("Error: Unable to create QA Agent.")

        with col2:
            if st.button("No"):
                create_agent_modal.close()
# st.write(st.session_state["directory_data"])
# st.write(st.session_state["agent_namespaces"])

with st.sidebar:
    with st.expander("📚 QA Agent", expanded=True):
        if st.session_state["agent"] is not None:
            for index, tool in enumerate(st.session_state["agent"].tools):
                st.write(f"**Tool name**: {tool.name}")
                st.write(f"**Tool description**: {tool.description}")
                if index != len(st.session_state["agent"].tools) - 1:
                    st.divider()

        else:
            st.write(None)

st.session_state["current_page"] = "knowledge_base"
