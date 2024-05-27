from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0)

from pathlib import Path
from langchain.document_loaders import TextLoader

doc_path = "E:\DEVELOP\LLM_Demo\examples\state_of_the_union.txt"

loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")

state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
###
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")

docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)

# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

res = agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)

print(res)

