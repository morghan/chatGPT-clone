from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from connections import vector_store
import streamlit as st


@st.cache_resource
def load_llm_and_memory():
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.2)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    print("Chat model and memory initialized")
    return chat_model, memory


chat_model, memory = load_llm_and_memory()


def create_qa_chain(
    template_prompt, chat_model=chat_model, memory=memory, vector_store=vector_store
):
    print("Creating QA chain")
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
