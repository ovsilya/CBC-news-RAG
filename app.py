from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import os
import uuid
import logging
import json
from tools import read_file, log_chat_history, get_session_history, setup_pinecone_and_tools
from utils import extract_metadata
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o'
)

# Set up retrievers and tools
retriever_news, retriever_guideline, tools = setup_pinecone_and_tools()

system_prompt = read_file('system_prompt.txt')

prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def chat(user_message: str):
    user_id = str(uuid.uuid4())
    
    try:
        result = agent_with_chat_history.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": user_id}}
        )
        response = result["output"]
        
        # Initialize response structure
        output = {
            "answer": response,
            "sources": extract_metadata(
                intermediate_steps=result.get("intermediate_steps", []),
                retriever_news=retriever_news,
                retriever_guideline=retriever_guideline,
                user_message=user_message
            )
        }
        
        log_chat_history(user_id)
        return json.dumps(output, indent=2)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        output = {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
        return json.dumps(output, indent=2)

if __name__ == "__main__":
    # List of example messages to test
    test_messages = [
        "What’s CBC’s guideline on citing anonymous sources?",
        "Suggest SEO headline for article: 1.6590078",
        "Summarize article in the tweet style: 1.6225449"
    ]
    
    for message in test_messages:
        print(f"\nTesting query: {message}\n")
        print(chat(message))