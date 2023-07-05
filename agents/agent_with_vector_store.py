from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from PyPDF2 import PdfReader
import pinecone
import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from conversation_chain import chatgpt_chain

namespaces = {
    "cookie_cutters": "Cookie Cutters",
    "bath_solutions": "Five Star Bath Solutions",
}
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["env"],
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def get_vectors_from_texts():
    cookie_cutters = PdfReader("files/Cookie Cutters Haircut for Kids - FDD.pdf")
    bath_solutions = PdfReader("files/Five Star Bath Solutions - FDD.pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )

    cookie_cutters_text = ""
    for page in cookie_cutters.pages:
        cookie_cutters_text += page.extract_text()

    cookie_cutters_chunks = text_splitter.split_text(text=cookie_cutters_text)

    bath_solutions_text = ""
    for page in bath_solutions.pages:
        bath_solutions_text += page.extract_text()

    bath_solutions_chunks = text_splitter.split_text(text=bath_solutions_text)

    print("cookie cutters", len(cookie_cutters_chunks))
    print("bath solutions", len(bath_solutions_chunks))

    return Pinecone.from_texts(
        texts=cookie_cutters_chunks,
        embedding=embeddings,
        index_name=st.secrets["pinecone"]["index_name"],
        namespace=namespaces["cookie_cutters"],
    ), Pinecone.from_texts(
        texts=bath_solutions_chunks,
        embedding=embeddings,
        index_name=st.secrets["pinecone"]["index_name"],
        namespace=namespaces["bath_solutions"],
    )


def get_qa_chains():
    qa_chains = []
    for namespace in list(namespaces.values()):
        vector_store = Pinecone.from_existing_index(
            index_name=st.secrets["pinecone"]["index_name"],
            embedding=embeddings,
            namespace=namespace,
        )
        qa_chains.append(
            RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            )
        )

    return qa_chains


qa_chains = get_qa_chains()


tools = [
    # Tool(
    #     name="Conversation bot",
    #     func=chatgpt_chain.run,
    #     description="Useful for when the user needs to have or follow a conversation. Input can be a fully formed question or any kind of sentence. Return the output directly to the user.",
    #     return_direct=True,
    # ),
    Tool(
        name="Cookie Cutters QA System",
        func=qa_chains[0].run,
        description="Useful for when you need to answer questions about the Cookie Cutters franchise. Input should be a fully formed question.",
    ),
    Tool(
        name="Five Star Bath Solutions QA System",
        func=qa_chains[1].run,
        description="Useful for when you need to answer questions about the Five Star Bath Solutions franchise. Input should be a fully formed question.",
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    #     agent_kwargs={
    #         "prefix": "You are an asistant whose only task is to take the user's request and decide the best tool to answer the user's request. Do not attempt to answer questions by yourself. Always use a tool. You have access to the following tools:",
    #         "format_instructions": """Use the following format:
    # Request: the input from the user
    # Thought: you should always think about what to do to handle the user's input
    # Action: the action to take, should be one of [Conversation bot, Cookie Cutters QA System, Five Star Bath Solutions QA System]
    # Action Input: the input to the action. If the action is the Conversation bot, this should be the user's input as it is.
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original request""",
    #     },
)

while True:
    query = input("💬 To exit type 'q', else Enter a query for GPT: ")
    if query == "q":
        break
    print("\n\n---------------------------\n")
    with get_openai_callback() as cb:
        response = agent.run(query)
        print(response)
        print(cb)
    print("\n---------------------------\n\n")

# print(get_vectors_from_texts())
