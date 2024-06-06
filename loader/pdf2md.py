from pathlib import Path
import aspose.words as aw
import os

import elasticsearch
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


def pdf2markdown(filepath):
    filepath = r'../../bs_challenge_financial_14b_dataset/pdf/'
    file_name = []
    path_list = []
    for i in os.listdir(filepath):
        file_name.append(i)
        path_list.append(os.path.join(filepath + i))

    for j in range(10):
        print("index" + str(j))
        doc = aw.Document(path_list[j])
        doc.save(r'../md_files/{}.md'.format(file_name[j].split('.')[0]))

        return doc


def es_addText(splits: list[Document], index_name: str):
    elasticsearch_url = "http://localhost:9200"
    docs = [doc.page_content for doc in splits]

    client = elasticsearch.Elasticsearch(elasticsearch_url)
    retriever = ElasticSearchBM25Retriever(client=client, index_name=index_name)
    retriever.add_texts(docs)


def md2FaissES():
    embeddings = HuggingFaceEmbeddings(model_name="D:/code_all/HuggingFace/bge")
    splitter = MarkdownTextSplitter(chunk_size=256, chunk_overlap=32)
    vector_store = FAISS.from_documents([Document(" ")], embeddings)

    directory = Path('../md_file')
    index_name = "md_faiss_es"
    # 遍历目录下所有.md文件
    for md_file in directory.glob('**/*.md'):
        print(md_file)
        loader = UnstructuredFileLoader(str(md_file))
        docs = loader.load()
        splits = splitter.split_documents(docs)

        vector_store.add_documents(splits)
        es_addText(splits, index_name)
    vector_store.save_local(index_name)


if __name__ == '__main__':
    md2FaissES()
