import os
import uuid
from pprint import pprint
from typing import Dict, Any
import datasets
from datasets import Dataset
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
from llama_index.evaluation import RetrieverEvaluator, generate_question_context_pairs
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall, context_precision
from loader.PDF_loader import RapidOCRPDFLoader

# from ragas.langchain.evalchain import RagasEvaluatorChain

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatTongyi()


# 文档划分 与 嵌入
def RAG_fun() -> RetrievalQA:
    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    vector_store = FAISS.load_local("loader/md_faiss_index_10", embeddings)
    retriever = vector_store.as_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever
    )

    return chain


def RAG_run(inputs) -> dict[str, Any]:
    # splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=100)
    # loader = UnstructuredFileLoader("./md_files/test.md")
    # docs = loader.load()
    # splits = splitter.split_documents(docs)
    # for i in splits:
    #     print(i)

    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    vector_store = FAISS.load_local("loader/md_faiss_index_10", embeddings)
    retriever = vector_store.as_retriever(k=5)

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    return chain.invoke(inputs)


def RAG_rerank(query: str, result: dict):
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    from llama_index.schema import NodeWithScore, QueryBundle, TextNode

    reranker = FlagEmbeddingReranker(
        top_n=3,
        model="D:/code_all/HuggingFace/bge-reranker-large",
        use_fp16=False
    )
    documents = [doc.page_content for doc in result["source_documents"]]
    pprint(documents)
    print("-" * 50)

    nodes = [NodeWithScore(node=TextNode(text=doc)) for doc in documents]

    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    for node in ranked_nodes:
        print(node.node.get_content(), "-> Score:", node.score)
        print("*" * 50)


def eval_fun(result):
    """原生Ragas方法，版本 >= 0.1.0"""
    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    dataset = Dataset.from_dict({
        "question": [result["query"]],
        "answer": [result["result"]],
        "contexts": [[result["source_documents"][0].page_content]]  # 这里设置为弟弟一个文档是因为evaluate只能对一个文档和query进行测评
    })

    eval_results = evaluate(metrics=[faithfulness, answer_relevancy, context_relevancy], llm=llm, dataset=dataset,
                            embeddings=embeddings)
    pprint(eval_results)

    """RagasEvaluatorChain的方法需要使用chatgpt，且 ragas 的版本<0.1.0"""
    # from langchain_together import Together
    # make eval chains
    # eval_chains = {
    #     m.name: RagasEvaluatorChain(llm=llm, metric=m)
    #     for m in [faithfulness, answer_relevancy, context_relevancy, context_recall]
    # }
    #
    # for name, eval_chain in eval_chains.items():
    #     score_name = f"{name}_score"
    #     print(f"{score_name}: {eval_chain(result)[score_name]}")


if __name__ == '__main__':
    query = "武汉力源信息技术股份有限公司的地址在哪里？"
    result = RAG_run(query)
    RAG_rerank(query, result)
    # eval_fun(result)
