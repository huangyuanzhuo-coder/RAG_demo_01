import os
from filetype.types import DOCUMENT
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain.output_parsers import format_instructions
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.tools import Tool

from loader.PDF_loader import RapidOCRPDFLoader

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["OPENAI_API_KEY"] = "sk-proj-O3Xl8o9FpCMw4ePxqweiT3BlbkFJ3uFDUcCAcCWad9KcAQyv"

llm = ChatTongyi()


def pdf2doc(file_path) -> [DOCUMENT]:
    """
    :param file_path: pdf 文档
    :return: doc
    """
    # file_path = "bs_challenge_financial_14b_dataset/pdf/0b46f7a2d67b5b59ad67cafffa0e12a9f0837790.PDF"
    # loader = RapidOCRPDFLoader(file_path=file_path)
    # docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    loader = UnstructuredFileLoader("test.txt")
    docs = loader.load()
    splits = text_splitter.split_documents(docs)

    # test
    # docs = retriever.get_relevant_documents("武汉力源信息技术股份有限公司的地址在哪里？")
    # print(docs)

    return splits


def RAG_fun() -> RetrievalQA:
    # 文档划分 与 嵌入

    splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=100)
    loader = UnstructuredFileLoader("./md_files/test.md")
    docs = loader.load()
    splits = splitter.split_documents(docs)
    for i in splits:
        print(i)

    embeddings = HuggingFaceEmbeddings(model_name="D:\code\HuggingFace/bge-large-zh-v1.5")
    # embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device='cpu')
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """基于以下已知信息，简洁和专业的回答用户的问题。
        如果无法从中得到答案，请说”根据已知信息无法回答该问题“ 或 ”没有提供足够的相关信息“。不允许在答案中添加编造成分，答案请使用中文。

        已知内容：
        {context}

        问题：
        {question}
    """

    # prompt = ChatPromptTemplate.from_template(
    #     prompt_template,
    #     partial_variables={"format_instructions": format_instructions},
    # )
    #
    # setup_and_retrieval = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # )
    # # RAG
    # chain = setup_and_retrieval | prompt | llm
    # return chain.invoke(inputs)

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever
    )

    return chain


if __name__ == '__main__':
    print(RAG_fun("武汉力源信息技术股份有限公司的地址在哪里？"))
