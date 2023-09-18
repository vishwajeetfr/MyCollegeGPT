import os
from apikey import apikey
import streamlit as st
import pinecone
# import nest_asyncio
# nest_asyncio.apply()
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA


os.environ['OPENAI_API_KEY'] = apikey
st.title('MyCollege gpt')

# initialize pinecone
pinecone.init(
    api_key='MyPincode_API_key',
    environment='MyPincode_Environment'
)

loader = SitemapLoader(
    "https://dypatilstadium.com/wp-sitemap-posts-post-1.xml",
    filter_urls=[
        "https://dypatilstadium.com/2017/10/26/fifa-u17-world-cup-2017-india/"]
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
)

docs_chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()

index_name = "MyIndexName"

# create a new index
docsearch = Pinecone.from_documents(
    docs_chunks, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)
# query = "What is the highest package recieved"
# docs = docsearch.similarity_search(query)
# print(docs[0])


# prompt = st.text_input('ask me here')


llm = OpenAI(temperature=0.9)


# if prompt:
# response = llm(prompt)
# st.write(response)

qa_with_sources = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

query = st.text_input('ask me here')
if query:
    result = qa_with_sources({"query": query})
    st.write(result["result"])
