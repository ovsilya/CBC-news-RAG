# editorials.py
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
import os
import requests
import time
import uuid
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PINECONE_REGION = os.environ["PINECONE_ENVIRONMENT"]
EDITORIAL_INDEX_NAME = os.environ["INDEX_GUIDELINE"]

def initialize_vector_store():
    """Initialize or connect to the Pinecone vector store."""
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if EDITORIAL_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index: {EDITORIAL_INDEX_NAME}")
            pc.create_index(
                name=EDITORIAL_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
            )
            # Wait for readiness
            while not pc.describe_index(EDITORIAL_INDEX_NAME).status["ready"]:
                logger.info(f"Waiting for index {EDITORIAL_INDEX_NAME} to be ready...")
                time.sleep(2)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=EDITORIAL_INDEX_NAME,
            embedding=embeddings
        )
        logger.info("Vector store initialized")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def get_dynamic_html(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    html = driver.page_source
    driver.quit()
    return html

def process_webpage(url, vector_store):
    """Extract, chunk, embed and store webpage content in Pinecone."""
    try:
        response = requests.get(url)
        html = get_dynamic_html(url)
        soup = BeautifulSoup(html, "html.parser")


        content_area = soup.find("main") or soup.find("div", class_="contentArea") or soup
        if not content_area:
            logger.warning("Main content not found")
            return

        documents = []
        elements = content_area.find_all(["h1", "h2", "h3", "p", "li"])
        current_section = None
        section_content = []

        for el in elements:
            if el.name in ["h1", "h2", "h3"]:
                # Save previous section
                if current_section and section_content:
                    content = " ".join(section_content).strip()
                    if content:
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "doc_id": str(uuid.uuid4()),
                                    "section_title": current_section,
                                    "source_url": url,
                                    "document_type": "editorial_guideline"
                                }
                            )
                        )
                        logger.info(f"Processed section: {current_section}")
                    section_content = []
                current_section = el.get_text(strip=True)

            elif el.name in ["p", "li"]:
                text = el.get_text(strip=True)
                if text:
                    section_content.append(text)

        # Save last section
        if current_section and section_content:
            content = " ".join(section_content).strip()
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "doc_id": str(uuid.uuid4()),
                            "section_title": current_section,
                            "source_url": url,
                            "document_type": "editorial_guideline"
                        }
                    )
                )

        if not documents:
            logger.warning("No valid sections extracted")
            return

        # Chunk & embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        logger.info(f"Uploading {len(chunks)} chunks to Pinecone index '{EDITORIAL_INDEX_NAME}'...")
        vector_store.add_documents(chunks)
        logger.info("Upload complete.")
    except Exception as e:
        logger.error(f"Error processing webpage: {e}")
        raise

def process_editorial_data_from_file(file_path):
    """Process multiple editorial guideline pages from a file of URLs."""
    try:
        with open(file_path, "r") as f:
            urls = list({line.strip() for line in f if line.strip()})
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print(f"[ERROR] File not found: {file_path}")
        return

    if not urls:
        logger.warning("No valid URLs found in file.")
        print("[WARNING] No valid URLs found in file.")
        return

    logger.info(f"Found {len(urls)} URLs to process.")
    print(f"[INFO] Found {len(urls)} URLs to process.")

    vector_store = initialize_vector_store()

    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{len(urls)}] Processing URL: {url}")
        try:
            process_webpage(url, vector_store)
            print(f"[SUCCESS] Finished processing: {url}")
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            print(f"[ERROR] Failed to process {url}: {e}")



if __name__ == "__main__":
    FILE_PATH = "pages.txt"
    process_editorial_data_from_file(FILE_PATH)
