# CBC News and Editorial Guidelines Chatbot

This project implements a chatbot that leverages AI to answer queries about CBC news articles and internal editorial guidelines (Journalistic Standards and Practices). The chatbot uses a combination of LangChain, OpenAI's GPT-4o model, and Pinecone vector stores to retrieve relevant information and provide structured JSON responses with metadata for downstream applications.

## Project Overview

The chatbot is designed to:
- Retrieve and summarize CBC news articles based on queries (e.g., article IDs or topics).
- Provide information on CBC's editorial guidelines for queries about journalistic standards.
- Return structured JSON output with the answer and metadata (e.g., content IDs, headlines, URLs, section titles).
- Maintain chat history for contextual conversations.
- Handle errors gracefully with detailed logging for debugging.

The codebase is modularized for readability and maintainability, with separate files for core logic (`app.py`), utility functions (`utils.py`), and tool/retriever setup (`tools.py`).

## Directory Structure

```
cbc-chatbot/
├── app.py                    # Main chatbot logic (agent execution and response formatting)
├── utils.py                  # Metadata extraction utilities
├── tools.py                  # Helper functions, Pinecone setup, vector stores, and retrievers
├── news_pinecone.py          # Script to upload news articles to the cbc-news Pinecone index
├── guidelines_pinecone.py    # Script to upload editorial guidelines to the cbc-editorial Pinecone index
├── system_prompt.txt         # System prompt for the LLM agent
├── news-dataset.json         # JSON file with news article data (required for news_pinecone.py)
├── pages.txt                 # Text file with guideline URLs (required for guidelines_pinecone.py)
├── assets/                   # Folder for static assets
│   └── demo.gif              # GIF showcasing the chatbot in action
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- A Pinecone account with API key
- An OpenAI account with API key
- Git installed for cloning the repository

### Installation
1. **Install Dependencies**
   Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Set your OpenAI and Pinecone API keys as environment variables. Replace the placeholders with your actual keys:
   ```bash
   export OPENAI_API_KEY="sk-proj-your-openai-key"
   export PINECONE_API_KEY="pcsk-your-pinecone-key"
   export PINECONE_ENVIRONMENT="us-east-1"
   export INDEX_NEWS="cbc-news"
   export INDEX_GUIDELINE="cbc-editorial"
   ```

3. **Prepare Input Files**
   - **News Data**: Use `news-dataset.json` file with news articles in the project root. Each article is a JSON object with at least `metadata` and `body`. Example:
     ```json
     [
       {
         "content_id": "1.6347842",
         "content_headline": "Want to make life more manageable? Miniature artists know the answer",
         "body": "Article text here...",
         "content_type": "article",
         "content_publish_time": "2022-10-01",
         "content_last_update": "2022-10-02",
         "content_word_count": "500",
         "content_department_path": "/news/canada",
         "content_categories": [{"content_category": "Art"}],
         "content_tags": [{"name": "miniature art"}]
       }
     ]
     ```
   - **Guideline URLs**:A `pages.txt` file created with URLs of CBC guideline pages. Example:
     ```
     https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/sources
     https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/investigative-journalism
     ```

4. **Upload Data to Pinecone**
   Populate the Pinecone indices before running the chatbot:
   - Upload news articles:
     ```bash
     python news_pinecone.py
     ```
     This processes `news-dataset.json`, chunks articles, and uploads them to the `cbc-news` index.
   - Upload editorial guidelines:
     ```bash
     python guidelines_pinecone.py
     ```
     This scrapes URLs from `pages.txt`, chunks content, and uploads to the `cbc-editorial` index.

5. **Run the Chatbot**
   Execute the main script to test the chatbot:
   ```bash
   python app.py
   ```
   The script runs a sample query: "Summarize this article for a Twitter post: 1.6590078". Modify the query in `app.py` to test other inputs.

## Technical Choices

### Language Model: GPT-4o
- **Why GPT-4o?**
  - **Advanced Reasoning**: GPT-4o, developed by OpenAI, excels at understanding complex queries and generating coherent, context-aware responses, making it ideal for a chatbot handling both news summaries and nuanced editorial guideline questions.
  - **Multimodal Capabilities**: While this project uses text-only features, GPT-4o's multimodal support (text, images) allows future enhancements (e.g., analyzing article images).
  - **Tool Integration**: GPT-4o integrates seamlessly with LangChain's tool-calling framework, enabling dynamic selection of retriever tools based on query type.
  - **Performance**: It balances speed and accuracy, providing quick responses for real-time interactions while maintaining high-quality outputs.

### Vector Store: Pinecone
- **Why Pinecone?**
  - **Scalability**: Pinecone is a managed vector database optimized for high-dimensional embeddings, handling large datasets like news articles and guidelines efficiently.
  - **Similarity Search**: Pinecone's similarity search (using cosine similarity) retrieves the most relevant documents based on query embeddings, crucial for accurate responses.
  - **Metadata Support**: Pinecone allows storing metadata (e.g., content IDs, headlines) with each vector, enabling structured retrieval for downstream applications.
  - **Ease of Use**: The `langchain-pinecone` integration simplifies vector store setup and querying within the LangChain ecosystem.

### Chunking Method
- **Approach**: Documents (news articles and guidelines) are chunked into smaller segments before embedding and storing in Pinecone.
- **Why Chunking?**
  - **Granularity**: Smaller chunks improve retrieval precision, ensuring the chatbot retrieves specific sections relevant to the query rather than entire documents.
  - **Embedding Quality**: OpenAI's `text-embedding-3-small` model generates better embeddings for shorter text segments, capturing semantic meaning more effectively.
  - **Metadata Attachment**: Each chunk is assigned metadata (e.g., `content_id`, `content_headline` for news; `source_url`, `section_title` for guidelines), ensuring every retrieved segment includes context for downstream use.
- **Implementation**: While not shown in the code (assumed to be handled during data ingestion), chunking is typically done using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of ~500-1000 characters and overlap of ~100 characters to preserve context.

### Separate Indices for News and Guidelines
- **Why Two Indices?**
  - **Content Differentiation**: News articles and editorial guidelines have distinct structures and purposes. News articles are time-sensitive and narrative-driven, while guidelines are static and policy-focused. Separate indices (`cbc-news` and `cbc-editorial`) allow tailored retrieval strategies.
  - **Query Optimization**: Using dedicated indices ensures the retriever targets the correct corpus (news or guidelines) based on the query, reducing noise and improving response accuracy.
  - **Metadata Specificity**: Each index stores metadata relevant to its content type (e.g., `content_id`, `content_headline` for news; `source_url`, `section_title` for guidelines), simplifying metadata extraction and response formatting.
  - **Scalability**: Separate indices enable independent scaling and maintenance, accommodating growth in either dataset without impacting the other.

### Metadata Integration
- **Approach**: Metadata is embedded with each chunk during data ingestion into Pinecone.
- **Why Per-Chunk Metadata?**
  - **Searchability**: Attaching metadata (e.g., `content_id`, `content_headline`) to each chunk allows searching by specific attributes (e.g., article ID) and enriches retrieved results with context.
  - **Granularity**: Per-chunk metadata ensures that even small text segments retain identifying information, critical for downstream applications that need source details.
  - **Consistency**: Every chunk in the `cbc-news` index includes `content_id` and `content_headline`, while every chunk in the `cbc-editorial` index includes `source_url` and `section_title`, guaranteeing metadata availability in responses.

### Project Modularization: `utils.py` and `tools.py`
- **Why Split into `utils.py` and `tools.py`?**
  - **Separation of Concerns**: `utils.py` handles metadata extraction, a distinct utility function, while `tools.py` manages helper functions, vector store setup, and retriever/tool definitions, aligning with their respective roles.
  - **Readability**: Modularizing reduces the size of `app.py`, making the core chatbot logic easier to follow.
  - **Maintainability**: Changes to metadata processing or retriever setup can be made in `utils.py` or `tools.py` without altering the main application.
  - **Reusability**: Functions in `utils.py` and `tools.py` can be reused in other projects or scripts.
- **Contents**:
  - `utils.py`: Contains `extract_metadata`, which processes intermediate steps from the agent to extract metadata from retrieved documents.
  - `tools.py`: Includes `read_file`, `log_chat_history`, `get_session_history`, and `setup_pinecone_and_tools` for initializing Pinecone and defining retrievers/tools.

### Metadata Extraction in `utils.py`
- **How It Works**:
  - The `extract_metadata` function in `utils.py` processes the `intermediate_steps` from the `AgentExecutor`, which contain tool calls and their outputs (retrieved documents or strings).
  - **Handling Document Outputs**: If the retriever tool returns a list of `Document` objects, the function extracts metadata (`content_id`, `content_headline` for news; `source_url`, `section_title` for guidelines) from each document's `metadata` attribute.
  - **Handling String Outputs**: If the retriever tool returns a string (due to LangChain's serialization), the function falls back to invoking the appropriate retriever (`retriever_news` or `retriever_guideline`) with the user query to fetch `Document` objects directly.
  - **Ensuring Metadata Availability**: By attaching metadata to each chunk during ingestion, every retrieved document includes metadata, ensuring the JSON response contains source details for downstream use.
  - **Error Handling**: The function logs warnings for unexpected output types (e.g., non-`Document` objects) and uses default empty strings for missing metadata fields, ensuring robustness.
- **Why Per-Chunk Metadata Matters**: Attaching metadata to chunks (not the entire document) ensures that even partial matches during retrieval include source context, critical for applications that need precise attribution (e.g., linking to specific articles or guideline sections).

## Usage
Run the chatbot with:
```bash
python app.py
```
The default query in `app.py` is "Summarize this article for a Twitter post: 1.6590078". Modify the `user_message` variable in `app.py` to test other queries, such as:
- News: "What’s the latest on Canadian politics?"
- Guidelines: "What’s CBC’s policy on anonymous sources?"

### Demo
Below is a short GIF demonstrating the chatbot in action, showing a sample query and its JSON response in the terminal.

![Chatbot Demo](assets/chatbot-demo.gif)

### Example Output
Below are example JSON outputs for specific queries, demonstrating the structured response format with metadata.

#### Query: "What’s CBC’s guideline on citing anonymous sources?"
```json
{
  "answer": "CBC's guidelines on citing anonymous sources emphasize the importance of verification and transparency. The guidelines state:\n\n1. **Use of Verified Information**: CBC stories are based on verified information, with a preference for first-hand, identifiable sources such as participants in an event or authenticated documents.\n\n2. **Conditions for Using Anonymous Sources**:\n   - Anonymous sources should be used exceptionally and must be approved by the Managing Editor.\n   - A confidential source is when the journalist knows the identity of the person but agrees to protect their identity.\n   - An anonymous source is when the identity of the source is not known to the journalist.\n\n3. **Verification Requirement**: Information obtained from confidential or anonymous sources should be verified before being released. CBC aims to corroborate the information with a second source, and in some cases, more than two sources may be required.\n\n4. **Editorial Oversight**: Any decision to publish a story based on a single confidential source must be referred to the Director.\n\n5. **Maintaining Independence**: Journalists must remain free from the influence of political, economic, or other interest groups, ensuring CBC’s independence is not compromised.\n\nThese guidelines ensure the integrity and credibility of CBC's journalistic practices.\n\nSource: [Citing Anonymous Sources], URL: [source_url]",
  "sources": [
    {
      "type": "guideline",
      "source_url": "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/use-of-social-media",
      "section_title": "Verification of User Generated Content (UGC) in News Stories"
    },
    {
      "type": "guideline",
      "source_url": "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/investigative-journalism",
      "section_title": "Verification of User Generated Content in News Stories"
    },
    {
      "type": "guideline",
      "source_url": "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/user-generated-content",
      "section_title": "Verification of User Generated Content (UGC) in news stories"
    },
    {
      "type": "guideline",
      "source_url": "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/sources",
      "section_title": "Verification of User Generated Content in News Stories"
    },
    {
      "type": "guideline",
      "source_url": "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/investigative-journalism",
      "section_title": "Protection of Sources – Granting Confidentiality"
    }
  ]
}
```

#### Query: "Suggest SEO headline for article: 1.6590078"
```json
{
  "answer": "Here is an SEO-optimized headline suggestion for the article:\n\n\"Discover Miniature Art: Transform Your Space, Transform Your Life\"\n\nThis headline is concise, keyword-rich, and emphasizes the transformative power of miniature art, which aligns with the article's theme. \n\nSource: [Want to make life more manageable? Miniature artists know the answer: sweat the small stuff](https://www.cbc.ca/news/canada/toronto/miniature-artists-1.6347842), ID: 1.6590078",
  "sources": [
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6609687",
      "content_headline": "Marketplace is back! We're celebrating our 50th anniversary: CBC's Marketplace cheat sheet"
    }
  ]
}
```

#### Query: "Summarize article in the tweet style: 1.6225449"
```json
{
  "answer": "New chief of Peguis First Nation, Stan Bird, raises questions about the flip of a prime property on Wellington Crescent. Despite its $4.9M valuation, it was sold for $350K and later resold for $2.8M. Bird seeks clarity on the transaction details and accounting. #PeguisFirstNation #RealEstate #Transparency\n\nSource: \"New chief of Peguis First Nation questions flip of desirable Wellington Crescent land,\" ID: 1.7114953",
  "sources": [
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    },
    {
      "type": "news",
      "content_id": "1.6347842",
      "content_headline": "Want to make life more manageable? Miniature artists know the answer: sweat the small stuff"
    }
  ]
}
```