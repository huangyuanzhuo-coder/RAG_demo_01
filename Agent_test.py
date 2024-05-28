import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import LLMMathChain
from langchain.output_parsers import pydantic

from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import Tool

from RAG_chat import RAG_fun

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["SERPAPI_API_KEY"] = "e7c902e28eb68593a62743a5b269f55ca6725996140fe6d8aff7408ad11ea118"  # google_key

llm = ChatTongyi()

tools = [
    Tool(
        name="Search_RAG",
        func=RAG_fun().run,
        description="根据问题在向量数据库中检索相关的信息,用于回答问题",
    )
]

"""
agent = llm_chain + output_parser + tools
llm_chain = llm + prompt
"""
# tools可以是外部工具，也可以是另外包装的llm_chain。例如llm-math就是LLMMathChain
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run("武汉力源信息技术股份有限公司的地址在哪里？")
print(result)
