from pprint import pprint
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import NodeWithScore, QueryBundle, TextNode


def RAG_rerank(query: str, docs: list[Document]) -> list[Document]:
    reranker = FlagEmbeddingReranker(
        top_n=3,
        model="D:/code_all/HuggingFace/bge-reranker-large",
        use_fp16=False
    )
    documents = [doc.page_content for doc in docs]
    nodes = [NodeWithScore(node=TextNode(text=doc)) for doc in documents]

    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    res = []
    for node in ranked_nodes:
        res.append(Document(node.node.get_content()))
        # print(node.node.get_content(), "-> Score:", node.score)
        # print("*" * 50)

    return res

class MixEsVectorRetriever(BaseRetriever):

    vector_retriever: BaseRetriever = None
    keyword_retriever: BaseRetriever = None
    combine_strategy: str = 'mix'

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # vector_docs = self.vector_retriever.get_relevant_documents(query,callbacks=run_manager.get_child())
        # keyword_docs = self.keyword_retriever.get_relevant_documents(query,callbacks=run_manager.get_child())
        #
        # combine_docs_dict = {}
        # min_len = min(len(vector_docs), len(keyword_docs))
        # for i in range(min_len):
        #     combine_docs_dict[keyword_docs[i].page_content] = keyword_docs[i]
        #     combine_docs_dict[vector_docs[i].page_content] = vector_docs[i]
        # for doc in keyword_docs[min_len:]:
        #     combine_docs_dict[doc.page_content] = doc
        # for doc in vector_docs[min_len:]:
        #     combine_docs_dict[doc.page_content] = doc
        #
        # print(combine_docs_dict)
        #
        # combine_docs = list(combine_docs_dict.values())
        # return combine_docs

        # EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(retrievers=[self.vector_retriever, self.keyword_retriever], weights=[0.5, 0.5])
        docs = ensemble_retriever.invoke(query)

        # rerank
        docs = RAG_rerank(query, docs)
        return docs

