from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="history", return_messages=True),
)

# output = chatgpt_chain.predict(
#     human_input="I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is pwd."
# )
# print(output)

# output = chatgpt_chain.predict(human_input="ls ~")
# print(output)

# while True:
#     query = input("💬 To exit type 'q', else Enter a query for GPT: ")
#     if query == "q":
#         break
#     print("\n\n---------------------------\n")
#     with get_openai_callback() as cb:
#         response = chatgpt_chain.run(query)
#         print(response)
#         print(cb)
#     print("\n---------------------------\n\n")
