from langchain_community.retrievers import ElasticSearchBM25Retriever

elasticsearch_url = "http://localhost:9200"
retriever = ElasticSearchBM25Retriever.create(elasticsearch_url, "langchain-index-4")