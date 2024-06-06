import elasticsearch
from langchain_community.retrievers import ElasticSearchBM25Retriever

elasticsearch_url = "http://localhost:9200"
index_name = "md_faiss_es_01"
docs = ["我昨天吃了好几种食物，比如牛奶、酸奶、披萨", "小明最喜欢吃三明治", "小明和小李经常出去玩", "你在说啥"]
query = "董事长是谁？"


client = elasticsearch.Elasticsearch(elasticsearch_url)
retriever = ElasticSearchBM25Retriever(client=client, index_name=index_name)

# retriever.add_texts(docs)

result = retriever.get_relevant_documents(query)
print(result)