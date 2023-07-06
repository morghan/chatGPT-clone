from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from connections import vector_store
import streamlit as st
import pinecone

embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["env"],
)
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


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
    with open("files/system-prompt.txt", "r") as file:
        sysprompt_content = file.read()

    template_prompt_content = template_prompt[0][2]
    full_prompt_content = template_prompt_content + "\n\n" + sysprompt_content

    system_prompt = PromptTemplate.from_template(template=full_prompt_content)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": system_prompt},
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        chain_type="stuff",
        verbose=True,
    )

    return qa_chain


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
