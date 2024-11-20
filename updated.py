import json
import logging
import os
import time

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer
from webdriver_manager.chrome import ChromeDriverManager

ask = "What was the status of the Philippines' external debt payment at the end of August?"
ask2 = "Why is the Philippines cautious about shifts in US foreign policy?"

# Set up Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = (
    "MY_API_TOKEN"
)

# Streamlit title for the web app
st.title("Business World Bot")

# Load the fine-tuned model and tokenizer
fine_tuned_model_path = "./fine_tuned_model1/final_checkpoint"
tokenizer_fine_tuned = BartTokenizer.from_pretrained(
    fine_tuned_model_path, local_files_only=True
)
fine_tuned_model1 = BartForConditionalGeneration.from_pretrained(
    fine_tuned_model_path
)

# Load the original BART model for summarization
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)

# Initialize HuggingFace Embedding Model
embedding_model = "multi-qa-mpnet-base-dot-v1"
base_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


def scrape_website(url):
    """
    Scrapes the specified URL using Selenium and extracts content and source URLs.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        list: A list of lists, each containing [content, source URL].
    """
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install())
        )
        driver.get(url)
        time.sleep(5)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        target_div = soup.find(
            "div",
            class_="td_block_wrap td_block_9 tdi_40 td-pb-border-top td_block_template_1",
        )

        if not target_div:
            logging.warning(f"Target div not found on the page: {url}")
            driver.quit()
            return []

        articles = []
        for link in target_div.find_all("a", href=True):
            if link["href"].startswith("http"):
                article_url = link["href"]
                driver.get(article_url)
                time.sleep(3)
                article_html = driver.page_source
                article_soup = BeautifulSoup(article_html, "html.parser")
                article_content = article_soup.get_text(separator=" ")
                articles.append([article_content, article_url])

        driver.quit()
        return articles

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []


def summarize_contexts(articles):
    """
    Summarizes a list of articles using the BART model.

    Args:
        articles (list): A list of lists, each containing [content, source URL].

    Returns:
        list: A list of lists, each containing [summarized content, source URL].
    """
    summarized_articles = []
    for article in articles:
        content, source = article
        inputs = tokenizer_bart(
            content, return_tensors="pt", max_length=512, truncation=True
        )
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=30,
            length_penalty=2.0,
        )
        summarized_content = tokenizer_bart.decode(
            summary_ids[0], skip_special_tokens=True
        )
        summarized_articles.append([summarized_content, source])
    return summarized_articles


def is_relevant_answer(query, response, threshold=0.7):
    """
    Checks the relevance of a generated response to a query using cosine similarity.

    Args:
        query (str): User query.
        response (str): Generated response.
        threshold (float): Cosine similarity threshold for relevance.

    Returns:
        bool: True if the response is relevant, False otherwise.
    """
    query_embedding = np.array(base_embeddings.embed_query(query))
    response_embedding = np.array(base_embeddings.embed_query(response))
    query_embedding = query_embedding.reshape(1, -1)
    response_embedding = response_embedding.reshape(1, -1)

    similarity_score = cosine_similarity(query_embedding, response_embedding)[
        0
    ][0]
    return similarity_score >= threshold


def generate_initial_answer(query):
    """
    Generates an initial answer using the fine-tuned model.

    Args:
        query (str): User query.

    Returns:
        str: Generated answer.
    """
    input_text = f"Query: {query} Context:"
    inputs = tokenizer_fine_tuned(
        input_text, return_tensors="pt", max_length=512, truncation=True
    )
    output_ids = fine_tuned_model1.generate(
        inputs["input_ids"], max_new_tokens=100
    )
    return tokenizer_fine_tuned.decode(output_ids[0], skip_special_tokens=True)


def generate_answer_with_fallback(query, retriever):
    """
    Generates an answer using the fine-tuned model with a fallback to contextual generation if needed.

    Args:
        query (str): User query.
        retriever: Retriever object for fetching relevant documents.

    Returns:
        str: Generated answer with cited sources.
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Use the most relevant retrieved document for context
    most_relevant_doc = retrieved_docs[0]

    context = most_relevant_doc.page_content  # Access content using attribute
    if not is_relevant_answer(query, context):
        # If no relevant documents are found, generate an answer from the fine-tuned model
        initial_answer = generate_initial_answer(query)
        if is_relevant_answer(query, initial_answer):
            return f"{initial_answer}\n\nSource: Answer derived from fine-tuned knowledge base."
        # return f"{initial_answer}\n\nSource: Answer derived from fine-tuned knowledge base."
        return "I'm sorry, I couldn't find a relevant answer to your query."

    source = most_relevant_doc.metadata.get(
        "source", "Unknown Source"
    )  # Access metadata using .metadata

    # Step 2: Generate the answer using the retrieved context
    input_text = f"Context: {context} Query: {query}"
    inputs = tokenizer_fine_tuned(
        input_text, return_tensors="pt", max_length=512, truncation=True
    )
    output_ids = fine_tuned_model1.generate(
        inputs["input_ids"], max_new_tokens=100
    )
    response = tokenizer_fine_tuned.decode(
        output_ids[0], skip_special_tokens=True
    )

    return f"{response}\n\nSource: {source}"


# Streamlit Input and Query Handling
url_input = st.text_input(
    "Enter URL to scrape data from", "https://www.bworldonline.com/"
)
query_input = st.text_input("Enter your query")
scrape_button = st.button("Scrape and Query")

if scrape_button:
    with st.spinner("Scraping website and processing data..."):
        # Step 1: Scrape the website
        articles = scrape_website(url_input)

        if articles:
            # Step 2: Summarize the scraped articles
            summarized_articles = summarize_contexts(articles)

            # Step 3: Split and add to vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=50
            )
            splits = []
            for summarized_article in summarized_articles:
                summarized_content, source = summarized_article
                chunks = text_splitter.split_text(summarized_content)
                for chunk in chunks:
                    splits.append([chunk, source])
            # print(splits)
            vectorstore = Qdrant.from_texts(
                texts=[split[0] for split in splits],
                metadatas=[{"source": split[1]} for split in splits],
                embedding=base_embeddings,
                collection_name="document_collection",
                location=":memory:",
            )
            st.success(
                "Data scraped, summarized, and added to the vector store!"
            )

        if query_input:
            # Step 4: Generate answer with fallback
            retriever = vectorstore.as_retriever()
            answer = generate_answer_with_fallback(query_input, retriever)
            st.write("Generated Answer:")
            st.write(answer)