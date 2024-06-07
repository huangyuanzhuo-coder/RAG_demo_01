import os
from pprint import pprint
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import Tool
from RAG_chat import RAG_fun, RAG_mix_fun

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["SERPAPI_API_KEY"] = "e7c902e28eb68593a62743a5b269f55ca6725996140fe6d8aff7408ad11ea118"  # google_key

llm = ChatTongyi()


def Chat_Agent():
    tools = [
        Tool(
            name="Search_company_info",
            func=RAG_mix_fun().run,
            description="根据用户的问题，在数据库中查询有关公司的信息，用于回答用户的问题",
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    """
    agent = llm_chain + output_parser + tools
    llm_chain = llm + prompt
    """
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)  # openai tools
    agent = initialize_agent(llm=llm,
                             tools=tools,
                             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                             memory=memory,
                             verbose=True
                             # agent_kwargs={"system_message": system_prompt.strip()}
                             )
    return agent


if __name__ == '__main__':
    agent = Chat_Agent()
    while True:
        print("-" * 60)
        query = input("\n用户：")
        if query.strip() == 'exit':
            break
        try:
            result = agent.run(query)
            pprint(result)
        except Exception as e:
            print("报错：", e)
            pass
