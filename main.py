from PyPDF2 import PdfReader
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

from connections import vector_store, fetch_system_prompt, upload_prompt
from stream_handlers import init, display_chat_history, submit
from langchain_handlers import create_qa_chain

template = fetch_system_prompt()

qa_chain = create_qa_chain(template)


def main(qa_chain=qa_chain):
    if "something" not in st.session_state:
        st.session_state["something"] = ""
    with st.sidebar:
        # System Message upload
        with st.form("system_prompt", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "You can customize the system prompt by uploading a .txt file. This prompt will be saved for future sessions",
                type=["txt"],
            )
            submitted1 = st.form_submit_button("Submit")
            if submitted1 and uploaded_file is not None:
                with st.spinner("Uploading and processing prompt 🚧 ..."):
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    upload_prompt(file_content)
                    template = fetch_system_prompt()
                    qa_chain = create_qa_chain(template)
                    st.session_state["chat_reset"] = True
                    if "chat_history" in st.session_state:
                        del st.session_state["chat_history"]

        # PDF file upload and storage in vector store
        with st.form("pdf_upload", clear_on_submit=True):
            pdf = st.file_uploader(
                "Upload your PDF. It may take some time, but it will be saved for future sessions",
                type=["pdf"],
            )
            submitted = st.form_submit_button("Submit")
        if submitted and pdf is not None:
            pdf_reader = PdfReader(pdf)
            st.success("✅ PDF file uploaded successfully")

            with st.spinner("Processing text 🚧 ..."):
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = text_splitter.split_text(text=text)
            st.success(f"✅ Text processed: {len(chunks)}")
            if vector_store is not None:
                with st.spinner("Adding text to vector store 🗄️ ..."):
                    try:
                        ids = vector_store.add_texts(texts=chunks, namespace="fdd")
                        st.success(f"✅ Text added successfully:")
                        st.write(ids)
                        # st.success("✅ Text added successfully")
                        st.session_state["chat_reset"] = True
                    except Exception as e:
                        st.write("Error:", e)

        st.text_input("Your input", key="user_input", on_change=submit)

    if uploaded_file is not None:
        st.success("File uploaded successfully. Here is your system prompt:", icon="📝")
        # st.code(uploaded_file.getvalue().decode("utf-8"), language="None")

    if st.session_state["something"] != "" and not st.session_state["chat_reset"]:
        with get_openai_callback() as callback, st.spinner("Thinking..."):
            ai_response = qa_chain(
                {
                    "question": st.session_state["something"],
                }
            )
            print(callback)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.session_state["chat_history"].append(
            (st.session_state["something"], ai_response["answer"])
        )

    display_chat_history()


if __name__ == "__main__":
    init()
    main()
