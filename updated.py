import logging
import os
import time

import numpy as np
import requests
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

# Set up Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = (
    "YOUR_HUGGINGFACE_API_TOKEN"
)

# Streamlit title for the web app
st.title("Business World Bot")

# Load the fine-tuned model and tokenizer for answer generation
fine_tuned_model_path = "./fine_tuned_model4/final_checkpoint"
tokenizer_fine_tuned = BartTokenizer.from_pretrained(
    fine_tuned_model_path, local_files_only=True
)
fine_tuned_model1 = BartForConditionalGeneration.from_pretrained(
    fine_tuned_model_path
)

# Load the original BART model for summarizing real-time data
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)

# Initialize HuggingFace Embedding Model for embedding
embedding_model = "multi-qa-mpnet-base-dot-v1"
base_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Initialize session state variables if they don't exist
if "is_scraped" not in st.session_state:
    st.session_state.is_scraped = False
    st.session_state.vectorstore = None
    st.session_state.retrieved_documents = []
    st.session_state.chat_history = []  # Stores the conversation history


# Function to scrape the website
def scrape_website(url):
    """
    Scrape the content and sub-links from a specific section of a webpage.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        tuple: A tuple containing:
            - content (str or None): The text content of the target `<div>` section, or `None` if not found.
            - sub_links (list): A list of URLs found within the target `<div>`.
    """

    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode
        # options.add_argument("--disable-gpu")  # Disable GPU acceleration
        # options.add_argument("--no-sandbox")  # Avoid sandboxing for better compatibility
        # options.add_argument("--disable-dev-shm-usage")  # Handle large data in shared memory
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
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


def validate_url(url):
    """
    Validates the URL by sending a HEAD request.
    Args:
        url (str): The URL to validate.
    Returns:
        bool: True if the URL is valid and accessible, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            return url
    except requests.RequestException:
        return False


# Summarize function
def summarize_context(context):
    """
    Generate a summary for the given text using a BART model.

    Args:
        context (str): The input text to summarize.

    Returns:
        str: A summarized version of the input text.
    """

    inputs = tokenizer_bart(
        context, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        no_repeat_ngram_size=2,
    )
    return tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)


# Generate initial answer
def generate_initial_answer(query):
    """
    Generate an initial answer to a given query using a fine-tuned language model.

    Args:
        query (str): The input query or question to process.

    Returns:
        str: The generated response based on the fine-tuned model.
    """

    input_text = f"Query: {query} Context:"
    inputs = tokenizer_fine_tuned(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )
    output_ids = fine_tuned_model1.generate(
        inputs["input_ids"], max_new_tokens=300, length_penalty=1.2
    )
    response = tokenizer_fine_tuned.decode(
        output_ids[0], skip_special_tokens=True
    )
    return response


# Check if answer is relevant
def is_relevant_answer(query, response, threshold=0.7):
    """
    Determine if a generated response is relevant to a given query based on cosine similarity.

    Args:
        query (str): The input query.
        response (str): The generated response to evaluate.
        threshold (float, optional): The similarity score threshold for relevance. Defaults to 0.7.

    Returns:
        bool: True if the similarity score is greater than or equal to the threshold, False otherwise.
    """

    query_embedding = np.array(base_embeddings.embed_query(query)).reshape(
        1, -1
    )
    response_embedding = np.array(
        base_embeddings.embed_query(response)
    ).reshape(1, -1)
    similarity_score = cosine_similarity(query_embedding, response_embedding)[
        0
    ][0]
    return similarity_score >= threshold


# Generate answer with fallback
def generate_answer_with_fallback(query, retriever):
    """
    Generate an answer to a query using a retrieval-based approach with a fallback mechanism.

    Args:
        query (str): The input query to answer.
        retriever (object): The retriever object used to fetch relevant documents.

    Returns:
        str: The generated answer, with source information if available, or an error message if no relevant answer is found.
    """

    retrieved_docs = retriever.get_relevant_documents(query)
    most_relevant_doc = retrieved_docs[0]
    context = most_relevant_doc.page_content
    if not is_relevant_answer(query, context):
        initial_answer = generate_initial_answer(query)
        if is_relevant_answer(query, initial_answer):
            answer, source = initial_answer.split("Source: ", 1)
            if validate_url(source):
                return f"{answer}\n\nSource: {source}"
        return "Could not find relevant answer"
    # Use the most relevant retrieved document for context
    source = most_relevant_doc.metadata.get("source", "Unknown Source")

    # Generate the answer using the retrieved context
    input_text = f"Context: {context} Query: {query}"
    inputs = tokenizer_fine_tuned(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )
    output_ids = fine_tuned_model1.generate(
        inputs["input_ids"], max_new_tokens=300, length_penalty=1.2
    )
    response = tokenizer_fine_tuned.decode(
        output_ids[0], skip_special_tokens=True
    )
    answer, _ = response.split("Source: ", 1)
    if validate_url(source):
        return f"{answer}\n\nSource: {source}"
    return answer, context


# Streamlit user inputs
url = "https://www.bworldonline.com/"
query_input = st.text_area(
    "Enter your query"
)  # Text area allows for multi-line input
send_button = st.button("Send")

# Display conversation history
st.write("**Conversation History:**")
for i, (user_message, bot_response) in enumerate(
    st.session_state.chat_history
):
    st.write(f"**User:** {user_message}")
    st.write(f"{bot_response}")

# When the user submits a message
if send_button and query_input.strip():
    with st.spinner("Processing..."):
        if not st.session_state.is_scraped:
            # Scrape website and summarize
            content, sub_links = scrape_website(url)
            if content:
                loader = WebBaseLoader(web_paths=sub_links, bs_kwargs=None)
                documents = loader.load()
                summarized_docs = [
                    summarize_context(doc.to_json()["kwargs"]["page_content"])
                    for doc in documents
                ]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512, chunk_overlap=50
                )
                splits = text_splitter.split_documents(documents)
                vectorstore = Qdrant.from_documents(
                    splits,
                    embedding=base_embeddings,
                    collection_name="document_collection",
                    location=":memory:",
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.retrieved_documents = splits
                st.session_state.is_scraped = True
                st.success("Data scraped and stored successfully!")

        # Set up retriever and answer query using conversational memory
        retriever = st.session_state.vectorstore.as_retriever()
        answer = generate_answer_with_fallback(query_input, retriever)

        # Add the current query and response to the chat history
        st.session_state.chat_history.append((query_input, answer))

        # Clear the input field after sending the message
        st.rerun()