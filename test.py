import os
import random
from pprint import pprint
from dashscope import Generation
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.tools import Tool

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"

llm = ChatTongyi()

response =


def say_again(input: str) -> int:
    return random.randint(0, 10)


system_prompt = "你是一个历史学家，请你以诸葛亮的口吻来回答用户的问题,使用文言文回答问题"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=system_prompt),

])

tools = [
    Tool(
        func=say_again,
        name="Say Again",
        description="A tool that repeats the input back to the user.",
    ),
]


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(llm=llm,
                               tools=tools,
                               verbose=True,
                               memory=memory,
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                               agent_kwargs={"system_message": system_prompt.strip()}
                               )
res = agent_chain.run(input="你好啊")
pprint(res)
responses = Generation.call(Generation.Models.qwen_max, messages=messages, result_format='message', stream=True,
                            incremental_output=True)


# from simpleaichat import AIChat
#
# ai = AIChat(client=llm, system="根据用户提供的项目名称创建一个精美的GitHub README。")
# ai("simpleaichat")