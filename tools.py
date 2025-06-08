from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import logging
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NEWS = os.environ["INDEX_NEWS"]
INDEX_GUIDELINE = os.environ["INDEX_GUIDELINE"]

# Initialize session history store
store = {}

def read_file(file_name):
    """Helper function to read a file's content."""
    try:
        with open(file_name, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File {file_name} not found.")
        return ""

def log_chat_history(session_id: str):
    """Log the chat history for a given session."""
    history = get_session_history(session_id)
    #logger.info(f"Chat history for session {session_id}: {history.messages}")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def setup_pinecone_and_tools():
    """Set up Pinecone vector stores, retrievers, and tools."""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Connect to Pinecone vector stores
    index_guideline = pc.Index(INDEX_GUIDELINE)
    index_news = pc.Index(INDEX_NEWS)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store_news = PineconeVectorStore(index=index_news, embedding=embeddings)
    vector_store_guideline = PineconeVectorStore(index=index_guideline, embedding=embeddings)

    # Define retrievers
    retriever_news = vector_store_news.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    retriever_guideline = vector_store_guideline.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    # Define tools with distinct names and descriptions
    retriever_tool_news = create_retriever_tool(
        retriever_news,
        "news_retriever",
        description="Use this tool to retrieve information from CBC news articles and content. Suitable for queries about articles, headlines, or summaries."
    )

    retriever_tool_guideline = create_retriever_tool(
        retriever_guideline,
        "guideline_retriever",
        description="Use this tool to retrieve information from CBC's internal editorial guidelines (Journalistic Standards and Practices). Suitable for queries about editorial policies."
    )

    return retriever_news, retriever_guideline, [retriever_tool_news, retriever_tool_guideline]