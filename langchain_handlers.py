from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st
import pinecone

embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["env"],
)
chat = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def create_qa_agent(namespaces):
    tools = []
    for namespace in namespaces:
        vector_store = Pinecone.from_existing_index(
            index_name=st.secrets["pinecone"]["index_name"],
            embedding=embeddings,
            namespace=namespace,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        )
        tools.append(
            Tool(
                name=f"{namespace} QA System",
                func=qa_chain.run,
                description=f"Useful for when you need to answer questions about the {namespace} franchise. Input should be a fully formed question.",
            ),
        )
    qa_agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return qa_agent
