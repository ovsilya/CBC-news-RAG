from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import json
import os
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = os.environ["INDEX_NEWS"]

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    logger.info("Pinecone client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise

def initialize_vector_store():
    """Initialize the Pinecone vector store."""
    try:
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # Adjust based on your embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                logger.info(f"Waiting for index {INDEX_NAME} to be ready...")
                time.sleep(1)
        index = pc.Index(INDEX_NAME)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info(f"Vector store initialized for index: {INDEX_NAME}")
        return PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def process_json_file(json_file_path, vector_store):
    """Process a JSON file, extract text, and add it to the vector store."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON file: {json_file_path}")

        documents = []
        for item in data:
            # Validate required fields
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid item: not a dictionary")
                continue
            if "content_id" not in item or not item["content_id"]:
                logger.warning(f"Skipping item with missing or empty content_id")
                continue
            if "content_headline" not in item or not item["content_headline"].strip():
                logger.warning(f"Skipping item with missing or empty headline: ID={item.get('content_id')}")
                continue
            if "body" not in item or not item["body"].strip():
                logger.warning(f"Skipping item with empty or whitespace-only body: ID={item['content_id']}, Headline='{item['content_headline']}'")
                continue

            logger.info(f"Processing item: ID={item['content_id']}, Headline='{item['content_headline']}'")
            # Use only body for initial content
            content = item["body"]
            # Create metadata dictionary with safe defaults, including content_headline
            metadata = {
                "content_id": item["content_id"],
                "content_headline": item["content_headline"],
                "content_type": item.get("content_type", "Unknown"),
                "content_publish_time": item.get("content_publish_time", ""),
                "content_last_update": item.get("content_last_update", ""),
                "content_word_count": item.get("content_word_count", "0"),
                "content_department_path": item.get("content_department_path", "") if item.get("content_department_path") is not None else "",
                "content_categories": [cat["content_category"] for cat in item.get("content_categories", []) if isinstance(cat, dict) and "content_category" in cat],
                "content_tags": [tag["name"] for tag in item.get("content_tags", []) if isinstance(tag, dict) and "name" in tag]
            }
            documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            logger.warning("No valid documents to process")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        
        # Modify each chunk to prepend content_id and content_headline
        modified_docs = []
        for doc in split_docs:
            content_id = doc.metadata["content_id"]
            content_headline = doc.metadata["content_headline"]
            new_content = f"Content ID: {content_id}\nHeadline: {content_headline}\n{doc.page_content}"
            modified_docs.append(Document(page_content=new_content, metadata=doc.metadata))
        
        logger.info(f"Adding {len(modified_docs)} document chunks to vector store")
        vector_store.add_documents(modified_docs)
        logger.info("Finished adding document chunks to vector store")
    except Exception as e:
        logger.error(f"Error processing JSON file: {str(e)}")
        raise

def process_news_data(json_file_path):
    """Process the news dataset JSON file."""
    try:
        vector_store = initialize_vector_store()
        logger.info(f"Processing JSON file: {json_file_path}")
        process_json_file(json_file_path, vector_store)
        logger.info("Processing complete.")
    except Exception as e:
        logger.error(f"Error in process_news_data: {str(e)}")
        raise

if __name__ == "__main__":
    # Path to the local JSON file
    JSON_FILE_PATH = "news-dataset.json"
    process_news_data(JSON_FILE_PATH)