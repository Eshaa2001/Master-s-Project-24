import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer
from webdriver_manager.chrome import ChromeDriverManager

# Configuration Constants
HUGGING_FACE_TOKEN = "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"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
WEBSITE_URL = "https://www.bworldonline.com/"
SCRAPE_TIMEOUT = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


class BusinessWorldBot:
    """A conversational bot that scrapes business articles, generates responses, and manages query contexts."""

    def __init__(self):
        """Initialize the bot, loading models and setting up session state."""
        self._init_models()
        self._init_session_state()

    def _init_models(self):
        """Initialize the required models and embeddings with proper error handling."""
        try:
            # Fine-tuned model for answer generation
            fine_tuned_model_path = "C:/Users/eshaa/OneDrive/Documents/MFP/fine_tuned_model4/fine_tuned_model4/final_checkpoint"
            self.tokenizer_fine_tuned = BartTokenizer.from_pretrained(
                fine_tuned_model_path, local_files_only=True
            )
            self.fine_tuned_model = BartForConditionalGeneration.from_pretrained(
                fine_tuned_model_path
            )

            # Summarization model
            self.tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.bart_model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large-cnn"
            )

            # Embedding model
            self.base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            st.error("Failed to load models. Please check your configuration.")

    def _init_session_state(self):
        """Initialize or reset Streamlit session state."""
        if "is_scraped" not in st.session_state:
            st.session_state.update(
                {
                    "is_scraped": False,
                    "vectorstore": None,
                    "retrieved_documents": [],
                    "chat_history": [],
                }
            )

    @staticmethod
    def _configure_webdriver() -> webdriver.Chrome:
        """Configure Chrome WebDriver for scraping.

        Returns:
            webdriver.Chrome: Configured WebDriver instance.
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")

        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

    def scrape_website(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Scrape the specified website to extract content and sub-links.

        Args:
            url (str): The URL of the website to scrape.

        Returns:
            Tuple[Optional[str], List[str]]: Scraped text content and list of sub-links.
        """
        try:
            with self._configure_webdriver() as driver:
                driver.get(url)
                time.sleep(SCRAPE_TIMEOUT)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                target_div = soup.find(
                    "div",
                    class_="td_block_wrap td_block_9 tdi_40 td-pb-border-top td_block_template_1",
                )

                if not target_div:
                    logging.warning(f"Target div not found on the page: {url}")
                    return None, []

                sub_links = [
                    link["href"]
                    for link in target_div.find_all("a", href=True)
                    if link["href"].startswith("http")
                ]
                content = target_div.get_text(separator=" ")

                return content, sub_links
        except Exception as e:
            logging.error(f"Scraping error for {url}: {e}")
            return None, []

    def summarize_context(self, context: str) -> str:
        """Summarize the given context using the BART summarization model.

        Args:
            context (str): The input text to summarize.

        Returns:
            str: Summarized text.
        """
        try:
            inputs = self.tokenizer_bart(
                context, return_tensors="pt", max_length=512, truncation=True
            )
            summary_ids = self.bart_model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=50,
                length_penalty=2.0,
                no_repeat_ngram_size=2,
            )
            return self.tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Summarization error: {e}")
            return context[:300]  # Fallback to truncated context

    def is_relevant_answer(
        self, query: str, response: str, threshold: float = 0.7
    ) -> bool:
        """Check if the generated response is relevant to the query.

        Args:
            query (str): User query.
            response (str): Generated response.
            threshold (float, optional): Similarity threshold. Defaults to 0.7.

        Returns:
            bool: True if relevant, False otherwise.
        """
        try:
            query_embedding = self.base_embeddings.embed_query(query)
            response_embedding = self.base_embeddings.embed_query(response)
            similarity = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(response_embedding).reshape(1, -1),
            )[0][0]
            return similarity >= threshold
        except Exception as e:
            logging.error(f"Relevance check error: {e}")
            return False

    def validate_url(self, url: str) -> bool:
        """Validate the URL by sending a HEAD request.

        Args:
            url (str): URL to validate.

        Returns:
            bool: True if the URL is accessible, False otherwise.
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def output(self, query: str, context: str = "") -> str:
        """Generate a response for the given query and context.

        Args:
            query (str): User query.
            context (str, optional): Context text. Defaults to "".

        Returns:
            str: Generated response.
        """
        input_text = f"Query: {query} Context: {context}"
        inputs = self.tokenizer_fine_tuned(
            input_text, return_tensors="pt", max_length=1024, truncation=True
        )
        output_ids = self.fine_tuned_model.generate(
            inputs["input_ids"], max_new_tokens=300, length_penalty=1.2
        )
        return self.tokenizer_fine_tuned.decode(output_ids[0], skip_special_tokens=True)

    def generate_answer(self, query: str, retriever) -> str:
        """Generate an answer for the user's query using retrieved documents.

        Args:
            query (str): User query.
            retriever: Document retriever instance.

        Returns:
            str: Generated answer with source or an error message.
        """
        try:
            retrieved_docs = retriever.get_relevant_documents(query)
            context = retrieved_docs[0].page_content

            if not self.is_relevant_answer(query, context):
                initial_answer = self.output(query)
                if self.is_relevant_answer(query, initial_answer):
                    answer, source = initial_answer.split("Source: ", 1)
                    if self.validate_url(source):
                        return f"{answer}\n\nSource: {source}"
                return "Could not find relevant answer"

            source = retrieved_docs[0].metadata.get("source", "Unknown Source")
            final_answer = self.output(query, context)
            answer, _ = final_answer.split("Source: ", 1)
            return f"{answer}\n\nSource: {source}"
        except Exception as e:
            logging.error(f"Answer generation error: {e}")
            return "Unable to generate an answer. Please try again."

    def run(self):
        """Main Streamlit application logic for the bot."""
        st.title("Business World Bot")

        query_input = st.text_area("Enter your query")
        send_button = st.button("Send")

        # Display conversation history
        st.write("**Conversation History:**")
        for user_message, bot_response in st.session_state.chat_history:
            st.write(f"**User:** {user_message}")
            st.write(f"{bot_response}")

        if send_button and query_input.strip():
            with st.spinner("Processing..."):
                # Initial scraping if not done
                if not st.session_state.is_scraped:
                    content, sub_links = self.scrape_website(WEBSITE_URL)
                    if content:
                        loader = WebBaseLoader(web_paths=sub_links, bs_kwargs=None)
                        documents = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                        )
                        splits = text_splitter.split_documents(documents)

                        vectorstore = Qdrant.from_documents(
                            splits,
                            embedding=self.base_embeddings,
                            collection_name="document_collection",
                            location=":memory:",
                        )

                        st.session_state.vectorstore = vectorstore
                        st.session_state.retrieved_documents = splits
                        st.session_state.is_scraped = True
                        st.success("Data scraped and stored successfully!")

                # Generate answer
                retriever = st.session_state.vectorstore.as_retriever()
                answer = self.generate_answer(query_input, retriever)

                # Update chat history
                st.session_state.chat_history.append((query_input, answer))
                st.rerun()


def main():
    """Initialize and run the bot."""
    bot = BusinessWorldBot()
    bot.run()


if __name__ == "__main__":
    main()
