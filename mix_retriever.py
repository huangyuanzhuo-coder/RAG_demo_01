from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MixEsVectorRetriever(BaseRetriever):

    vector_retriever = BaseRetriever


    keyword_retriever = BaseRetriever
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
        ensemble_retriever = EnsembleRetriever(retriever=[self.vector_retriever, self.keyword_retriever], weights=[0.5, 0.5])
        docs = ensemble_retriever.invoke(query)

        return docs

