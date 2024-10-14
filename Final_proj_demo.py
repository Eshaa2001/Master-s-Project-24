from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant


minilm_embd: str = "all-MiniLM-L6-v2"
multi_qa_embd: str = "multi-qa-mpnet-base-dot-v1"
avso_embd: str = "avsolatorio/GIST-Embedding-v0"


def scrape_website(url):
    """
    Scrapes the website and extracts textual data from a specific div. It also collects URLs found in sub-links.
    
    Args:
        url (str): The URL of the website to scrape.

    Returns:
        tuple: A tuple containing the main content as a string and a list of URLs found in sub-links.
    """
    # Setting up Selenium WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    
    # The website is highly dynamic, adding a delay to let content load
    time.sleep(5) 
    
    # Get the main page content
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    # Targeting the specific div with the class 'td_block_wrap td_block_9 tdi_40 td-pb-border-top td_block_template_1'
    target_div = soup.find('div', class_='td_block_wrap td_block_9 tdi_40 td-pb-border-top td_block_template_1')
    
    if not target_div:
        print("Target div not found on the page.")
        driver.quit()
        return None, None

    # Extract all links within this div 
    sub_links = target_div.find_all('a', href=True)
    
    # Collect URLs and content
    urls = [link['href'] for link in sub_links if link['href'].startswith('http')]  
    all_content = ""

    # Extract main content from the target div (just text, no images)
    all_content += "Main Div Content:\n"
    all_content += target_div.get_text(separator=' ') + "\n\n"

    driver.quit()
    
    return all_content, urls


# Example 
url = 'https://www.bworldonline.com/'
web_content, web_links = scrape_website(url)
print(web_content)
print(web_links)


# Initializing the embedding model using HuggingFace's MiniLM model
base_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Embedding the web content
text = web_content

# Generating the query embedding
query_result = base_embeddings.embed_query(text)
print(f"Embedding dimension of the query result: {len(query_result)}")

# Embedding multiple documents (including the web content)
doc_result = base_embeddings.embed_documents([text, "This is not a test document."])

# Checking the number of document embeddings
print(f"Number of documents embedded: {len(doc_result)}")

# Getting embedding dimensions for each document
for i, embedding in enumerate(doc_result):
    print(f"Document {i} embedding dimension: {len(embedding)}")


_, web_links = scrape_website('https://www.bworldonline.com/')


if not web_links:
    print("No valid sub-links found.")
else:
    # Initialize WebBaseLoader with the scraped web links
    loader = WebBaseLoader(
        web_paths=web_links,
        bs_kwargs=None
    )

# Load the documents from the provided URLs
documents = loader.load()
doc_content = documents[0].to_json()["kwargs"]["page_content"]
print(len(doc_content.strip()))


# Initializing the text splitter with a chunk size of 128 characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
splits = text_splitter.split_documents(documents)
print("Number of splits/chunks: ", str(len(splits)))


# Displaying the content of a specific split (40th chunk)
print(splits[39].page_content)


# Storing the document embeddings in memory using Qdrant vector store
vectorstore = Qdrant.from_documents(
    splits,
    base_embeddings,
    location=":memory:",  # storing it locally in memory
    collection_name="test",
)

# Initializing a retriever from the vector store
retriever = vectorstore.as_retriever()

# Checking for similarity search using the query embedding
query = "Is rehabilitation of BNPP feasible?"
docs = vectorstore.similarity_search_by_vector(
    base_embeddings.embed_query(query)
)

# Print the retrieved documents
print(docs)
