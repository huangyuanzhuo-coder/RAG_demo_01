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
    retriever = vector_store.as_retriever(k=2)

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    return chain.invoke(inputs)


def RAG_evaluate(question):
    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    # embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device='cpu')
    vector_store = FAISS.load_local("loader/md_faiss_index_10", embeddings)
    retriever = vector_store.as_retriever()
    docs = vector_store.similarity_search_with_score(question, k=5)
    print(docs)

    querys = {}
    question_id = str(uuid.uuid4())
    querys[question_id] = question
    # qa_datasets = EmbeddingQAFinetuneDataset(
    #     queries=querys, corpus=node_dict, relevant_docs=relevant_docs
    # )
    # qa_dataset = generate_question_context_pairs(
    #     nodes, llm=llm, num_questions_per_chunk=2
    # )

    metrics = ["mrr", "hit_rate"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
    result = retriever_evaluator.evaluate(question, docs)


def eval_fun(result):
    """原生Ragas方法，版本 >= 0.1.0"""
    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    dataset = Dataset.from_dict({
        "question": [result["query"]],
        "answer": [result["result"]],
        "contexts": [[result["source_documents"][0].page_content]]
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
    result = RAG_run("武汉力源信息技术股份有限公司的地址在哪里？")
    eval_fun(result)
