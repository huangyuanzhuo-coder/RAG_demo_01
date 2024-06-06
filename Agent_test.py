import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import Tool
from RAG_chat import RAG_fun, RAG_mix_fun
from langchain.agents.react.base import DocstoreExplorer
os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["SERPAPI_API_KEY"] = "e7c902e28eb68593a62743a5b269f55ca6725996140fe6d8aff7408ad11ea118"  # google_key

llm = ChatTongyi()

tools = [
    Tool(
        name="Search_company_info",
        func=RAG_mix_fun().run,
        # description="从各个公司的招股说明书中检索相关内容，用于回答用户的问题",
        description="根据用户的问题，在数据库中查询有关公司的信息，用于回答用户的问题",
    )
]

"""
agent = llm_chain + output_parser + tools
llm_chain = llm + prompt
"""
# tools = load_tools(["serpapi", "llm-math"], llm=llm)  # openai tools
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# result = agent.run("安徽黄山胶囊股份有限公司的地址在哪里？")
# result = agent.run("安徽黄山胶囊股份有限公司的董事长是谁？")
result = agent.run("我想查询安徽黄山胶囊股份有限公司在2016年1-6月的营业收入是多少？")

print(result)



