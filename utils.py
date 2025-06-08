import logging
from langchain_core.documents import Document

# Set up logging
logger = logging.getLogger(__name__)

def extract_metadata(intermediate_steps, retriever_news, retriever_guideline, user_message):
    """
    Extract metadata from intermediate steps of an AgentExecutor run.
    
    Args:
        intermediate_steps: List of (AgentAction, observation) tuples from AgentExecutor.
        retriever_news: LangChain retriever for news articles.
        retriever_guideline: LangChain retriever for editorial guidelines.
        user_message: The user's input query for fallback retriever calls.
    
    Returns:
        List of source metadata dictionaries with type, content_id, content_headline, source_url, or section_title.
    """
    sources = []
    
    for step in intermediate_steps:
        action = step[0]  # AgentAction
        observation = step[1]  # Tool output
        tool = action.tool
        
        # Log the observation for debugging
        # logger.info(f"Tool used: {tool}, Observation type: {type(observation)}, Observation: {observation}")
        
        # Handle different observation types
        if isinstance(observation, list):
            # Expected case: observation is a list of Document objects
            for doc in observation:
                if isinstance(doc, Document):
                    metadata = doc.metadata or {}
                    if tool == "news_retriever":
                        sources.append({
                            "type": "news",
                            "content_id": metadata.get("content_id", ""),
                            "content_headline": metadata.get("content_headline", "")
                        })
                    elif tool == "guideline_retriever":
                        sources.append({
                            "type": "guideline",
                            "source_url": metadata.get("source_url", ""),
                            "section_title": metadata.get("section_title", "")
                        })
                else:
                    logger.warning(f"Unexpected document type: {type(doc)}")
        elif isinstance(observation, str):
            # Fallback: observation is a string, try to fetch documents directly from retriever
            logger.warning(f"Observation is a string, attempting to fetch documents directly")
            if tool == "news_retriever":
                docs = retriever_news.invoke(user_message)
                for doc in docs:
                    if isinstance(doc, Document):
                        metadata = doc.metadata or {}
                        sources.append({
                            "type": "news",
                            "content_id": metadata.get("content_id", ""),
                            "content_headline": metadata.get("content_headline", "")
                        })
                    else:
                        logger.warning(f"Unexpected retriever output type: {type(doc)}")
            elif tool == "guideline_retriever":
                docs = retriever_guideline.invoke(user_message)
                for doc in docs:
                    if isinstance(doc, Document):
                        metadata = doc.metadata or {}
                        sources.append({
                            "type": "guideline",
                            "source_url": metadata.get("source_url", ""),
                            "section_title": metadata.get("section_title", "")
                        })
                    else:
                        logger.warning(f"Unexpected retriever output type: {type(doc)}")
        else:
            logger.warning(f"Unexpected observation type: {type(observation)}")
    
    return sources