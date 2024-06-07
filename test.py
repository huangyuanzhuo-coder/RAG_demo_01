import os
import random
from pprint import pprint

import elasticsearch
from dashscope import Generation
from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.tools import Tool

import mix_retriever
from RAG_chat import RAG_mix_fun, EMBEDDING_PATH, EMBEDDING_DEVICE

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"

llm = ChatTongyi()



def say_again(input: str) -> int:
    return random.randint(0, 10)


system_prompt = "你是一个历史学家，请你以诸葛亮的口吻来回答用户的问题,使用文言文回答问题"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=system_prompt),

])



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH, model_kwargs={'device': EMBEDDING_DEVICE})
vector_store = FAISS.load_local(f"loader/faiss_index_10_mix", embeddings)
vector_retriever = vector_store.as_retriever(k=5)

# keyword_retriever
elasticsearch_url = "http://localhost:9200"
client = elasticsearch.Elasticsearch(elasticsearch_url)
keyword_retriever = ElasticSearchBM25Retriever(client=client, index_name="faiss_index_10_mix")

# mix
mix_retriever = mix_retriever.MixEsVectorRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever,
                                                   combine_strategy="mix")
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=mix_retriever, memory=memory, verbose=True)
# res = chain("安徽黄山胶囊股份有限公司的董事长是谁？")
# pprint(res)
#
# res = chain("其他董事会成员都有谁")
# pprint(res)

tools = [
    Tool(
        func=chain,
        name="Search and answer",
        description="Search the details of the company to answer the question",
    ),
]


agent_chain = initialize_agent(llm=llm,
                               tools=tools,
                               verbose=True,
                               memory=memory,
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                               # agent_kwargs={"system_message": system_prompt.strip()}
                               )
res = agent_chain.run(input="安徽黄山胶囊股份有限公司的董事长是谁？")