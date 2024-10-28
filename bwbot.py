import logging
import os
import time

import streamlit as st
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from transformers import pipeline
from webdriver_manager.chrome import ChromeDriverManager

# Set up Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = (
    "YOUR_HUGGING_FACE_API_TOKEN"
)

# Streamlit title
st.title("RAG-based Answer Generation on Real-Time Web Scraped Data")

# HuggingFace Embedding Model
embedding_model = "multi-qa-mpnet-base-dot-v1"
base_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# HuggingFace Generation Model
generation_model = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 100},
)


# Function to scrape website
def scrape_website(url):
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install())
        )
        driver.get(url)
        time.sleep(5)  # Adjust as needed for dynamic content

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        target_div = soup.find(
            "div",
            class_="td_block_wrap td_block_9 tdi_40 td-pb-border-top td_block_template_1",
        )

        if not target_div:
            logging.warning(f"Target div not found on the page: {url}")
            driver.quit()
            return None, []

        sub_links = [
            link["href"]
            for link in target_div.find_all("a", href=True)
            if link["href"].startswith("http")
        ]
        content = target_div.get_text(separator=" ")
        driver.quit()

        return content, sub_links

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return None, []


# Function to generate answer
def generate_answer(query, retriever):
    retrieved_docs = retriever.get_relevant_documents(query)
    context_chunks = [doc.page_content for doc in retrieved_docs[:7]]
    context = " ".join(context_chunks)

    response = generation_model(
        f"Summarize to answer the question:\nContext: {context}\nQuestion: {query}\nAnswer:"
    )
    return response


# Streamlit Interface for URL Input and Query
url = "https://www.bworldonline.com/"
url_input = st.text_input(
    "Enter URL to scrape data from", "https://www.bworldonline.com/"
)
query_input = st.text_input(
    "Enter your query",
    "What will be the impact of Donald Trump returning to power?",
)
scrape_button = st.button("Scrape and Query")

# Scrape and Process Data
if scrape_button:
    with st.spinner("Scraping website and processing data..."):
        content, sub_links = scrape_website(url)
        loader = WebBaseLoader(web_paths=sub_links, bs_kwargs=None)
        documents = loader.load()
        doc_content = documents[0].to_json()["kwargs"]["page_content"]
        if content:
            # documents = [content] + sub_links
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
            # Embed the documents
            # embeddings = base_embeddings.embed_documents(splits)

            # Initialize Qdrant collection with documents and embeddings
            vectorstore = Qdrant.from_documents(
                splits,
                embedding=base_embeddings,
                collection_name="document_collection",
                location=":memory:",
            )
            st.success("Data scraped and added to the vector store!")

        if query_input:
            retriever = vectorstore.as_retriever()
            answer = generate_answer(query_input, retriever)
            st.write("Generated Answer:", answer)
