from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from conversation_chain import chatgpt_chain
from langchain.callbacks import get_openai_callback

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class ConversationInput(BaseModel):
    """Inputs for get_current_stock_price"""

    input: str = Field(description="User input to the chatbot. Must be a string.")


class ConversationTool(BaseTool):
    name = "run_conversational_chain"
    description = """
        Useful for when the user needs to have or follow a conversation. 
        Input can be a fully formed question or any kind of sentence. 
        Return the output directly to the user.
        """
    args_schema: Type[BaseModel] = ConversationInput

    def _run(self, input: str):
        chatbot_response = chatgpt_chain.run(input)
        return {"response": chatbot_response}

    def _arun(self, input: str):
        raise NotImplementedError("get_current_stock_price does not support async")


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    ConversationTool(),
]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

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
