import os
from pprint import pprint

import elasticsearch
import openai
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from RAG_chat import EMBEDDING_PATH, EMBEDDING_DEVICE
from mix_retriever import MixEsVectorRetriever

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"


def get_retriever():
    # vector_retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH, model_kwargs={'device': EMBEDDING_DEVICE})
    vector_store = FAISS.load_local(f"loader/faiss_index_10_mix", embeddings)
    vector_retriever = vector_store.as_retriever(k=5)

    # keyword_retriever
    elasticsearch_url = "http://localhost:9200"
    client = elasticsearch.Elasticsearch(elasticsearch_url)
    keyword_retriever = ElasticSearchBM25Retriever(client=client, index_name="faiss_index_10_mix")

    # mix
    mix_retriever = MixEsVectorRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever,
                                         combine_strategy="mix")
    return mix_retriever


if __name__ == '__main__':
    messages = []
    while True:
        query = input("query: ")
        if query == 'exit':
            break

        # retriever = get_retriever()
        # docs = retriever.get_relevant_documents(query)
        # prompt = "".join([doc.page_content for doc in docs])
        prompt = "hhhh"

        message = f"请你根据下面的信息{prompt}判断是否跟这个问题{query} 有关，如果你觉得无关请告诉我无法根据提供的上下文回答'{query}'这个问题，简要回答即可，否则请根据{prompt}对{query}的问题进行回答"
        messages.append({'role': Role.USER, 'content': message})
        res_message = ''
        # try:
        response = Generation.call(Generation.Models.qwen_max,
                                   messages=messages,
                                   result_format="message",
                                   # stream=True,
                                   api_key="sk-146d6977be0b406fb18a4bb9c54d9cf0",
                                   )
        print("system:", end='')
        pprint(response)
        for res in response:
            res_message += response.output.choices[0]["message"]["content"]
            print(response.output.choices[0]["message"]["content"], end='')
        print()
        messages.append({'role': Role.SYSTEM, 'content': res_message})
        # except Exception as e:
        #     print("报错:", e)
        #     pass
