# 📰 IntelliSearch: Multi-Source AI Research Assistant

IntelliSearch is a **Streamlit-based AI research assistant** that aggregates information from multiple sources, including local articles, news, web search, Wikipedia, and Arxiv. It uses **LangChain** and **Groq LLMs** to provide concise, summarized, and accurate responses to research questions.

---

## 🚀 Features

- **Multi-source retrieval**: Pulls content from:
  - Local articles via URLs
  - News using NewsAPI
  - Web search using DuckDuckGo
  - Wikipedia
  - Arxiv research papers
- **Hybrid mode**: Combines all sources intelligently for comprehensive answers.
- **Semantic search**: Uses FAISS and HuggingFace MiniLM embeddings for local document retrieval.
- **LLM integration**: Leverages ChatGroq (Groq LLM) for summarization and question answering.
- **Streamlit interface**: Interactive and user-friendly dashboard for input and responses.
- **Environment variable support**: API keys for Groq, HuggingFace, and NewsAPI loaded securely.

---

## 🧩 Tech Stack

- **LLM Backend:** Groq LLM (`llama-3.3-70b-versatile`)  
- **Vector Store:** FAISS with HuggingFace MiniLM embeddings  
- **Frameworks & Tools:** LangChain, Streamlit, dotenv  
- **API Integrations:** NewsAPI, DuckDuckGoSearch, WikipediaAPI, ArxivAPI  
- **Python Libraries:** pickle, os, re, numpy

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/intellisearch.git
cd intellisearch
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate  # on Windows
source venv/bin/activate  # on macOS/Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Create a .env File
Create a .env file in the root folder and add your API keys:

ini
Copy code
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
NEWS_API_KEY=your_newsapi_key  # optional
🎯 Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
In the sidebar:

Enter API keys for Groq, HuggingFace, and NewsAPI

Input up to 3 URLs for local articles (optional)

Choose the information source mode:

Local Articles

Internet News

Web Search

Wikipedia

Arxiv

Hybrid

Enter your research question and get summarized answers based on the chosen sources.

📂 Project Structure
bash
Copy code
intellisearch/
├── app.py                 # Main Streamlit application
├── vector_index.pkl       # FAISS index for local articles
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables (API keys)
🔧 Modes of Operation
Local Articles: Uses your provided URLs only.

Internet News: Fetches news articles using NewsAPI.

Web Search: Queries DuckDuckGo for online information.

Wikipedia: Retrieves factual summaries from Wikipedia.

Arxiv: Fetches recent research papers.

Hybrid: Combines all sources for a comprehensive answer.

⚡ Future Improvements
Add streaming responses for real-time answers

Multi-language support for queries and answers

Deploy as a web app or desktop app with GUI enhancements

Integrate additional knowledge sources and APIs

🙏 Acknowledgements
LangChain

Groq LLM

HuggingFace Transformers

FAISS

NewsAPI

WikipediaAPI

Arxiv API

Streamlit

💡 Author: [Vinay Lakkimsetti]
📅 Year: 2025
🔗 Repository: https://github.com/vinaylakkimsetti458/IntelliSearch
