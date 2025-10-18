import os
import re
import pickle
import streamlit as st
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain.tools import DuckDuckGoSearchResults

# Set up Streamlit page config
st.set_page_config(page_title="Multi-Source AI Research Assistant", page_icon="📰", layout="wide")
st.title("📰 IntelliSearch : Multi-Source AI Research Assistant")
st.sidebar.title("🔧 **Configuration Panel**")

# Sidebar for user inputs
with st.sidebar:
    st.markdown("### 🔑 **API Keys**")

    # Initializing Groq API Key
    groq_api_key = st.text_input(
        "Groq API Key",
        placeholder="Enter your Groq API key",
        type="password")
    
    # Initializing HuggingFace API Token
    huggingfacehub_api_token = st.text_input(
        "HuggingFace API Token",
        placeholder="Enter your HuggingFace API token",
        type="password")
    
    # Initializing News API Key
    news_api_key = st.text_input(
        "NewsAPI Key (optional)",
        placeholder="Enter your NewsAPI key",
        type="password")
    
    # Validation check for API keys
    if not groq_api_key or not huggingfacehub_api_token:
        st.warning("⚠️ Please enter Groq,HuggingFace and News API keys to proceed.")
        st.stop()
    else:
        st.success("✅ API keys loaded successfully!")

    # Load the API Key's into environment variables
    os.environ['GROQ_API_KEY'] = groq_api_key
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token

    # Horizontal divider
    st.markdown("---")

    # Section for URL input
    st.markdown("### 🌐 Article URLs")
    st.markdown("Enter up to **3 article URLs** for processing.")

    urls = []

    # Providing the urls
    for i in range(3):
        url = st.text_input(f"URL {i+1}", placeholder=f"Enter article URL {i+1}")
        if url:
            urls.append(url)

    # Submit button for processing URLs
    url_button = st.button("🚀 Process URLs")

    st.markdown("---")

    st.markdown("### 🧭 Information Source Mode")

    # Choosing the mode
    mode = st.radio(
        "Choose where to get information from:",
        ["Local Articles", "Internet News", "Web Search", "Wikipedia", "Arxiv", "Hybrid"]
    )

    st.markdown("---")

    # Footer Instructions
    st.markdown("""
    📝 Instructions:
    - Enter API keys first.
    - Add up to 3 article URLs.
    - Choose your mode:
      - Local Articles: Uses your provided URLs only.
      - Internet News: Uses NewsAPI.
      - Web Search: Uses DuckDuckGo web search.
      - Wikipedia: Fetches factual summaries.
      - Arxiv: Fetches recent research papers.
      - Hybrid: Combines all sources intelligently.
    """)
    
# Initializing LLM using ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_tokens=512,
    api_key=groq_api_key
)

# Initializing Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Initializing Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper()
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Initializing DuckDuckGoSearch Tool
duck_tool = DuckDuckGoSearchResults()

# Function to process URLs and create a FAISS index
def process_urls(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '.', ','],
            chunk_size=256)
        docs = text_splitter.split_documents(data)

        # Create embeddings and FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_index = FAISS.from_documents(docs, embedding=embeddings)

        # Save the index to a pickle file
        with open('vector_index.pkl', 'wb') as f:    
            pickle.dump({
                'index': vector_index.index,
                'docstore': vector_index.docstore,
                'index_to_docstore_id': vector_index.index_to_docstore_id
            }, f)
        return True
    
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")
        return False
    
# Handle URL button click
if url_button and urls:
    if process_urls(urls):
        st.success("✅ URLs processed and FAISS index created successfully!")
    else:
        st.warning("⚠️ Failed to process URLs.")

# Function to extract only the answer from the final result
def extract_answer(final_answer):
    return final_answer.strip()

# Function to get answer based on the query
def get_local_answer(query):
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load saved FAISS index
        with open('vector_index.pkl', 'rb') as f:
            saved_data = pickle.load(f)

        # Initialize FAISS store 
        faiss_store = FAISS(
            index=saved_data['index'],
            docstore=saved_data['docstore'],
            index_to_docstore_id=saved_data['index_to_docstore_id'],
            embedding_function=embeddings.embed_query
        )

        # Initialize Retrieval Chain
        retriever = faiss_store.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever)
        
        # Get the result from the retrieval chain
        result = chain.run(query)

        # Get the clean answer from the result
        return extract_answer(result)
    
    except Exception as e:
        return f"Error retrieving local answer: {str(e)}"

# Function to get News based on user query
def get_news(query, max_results=5):
    try:
        # Validation check for API Key
        if not news_api_key:
            return "NewsAPI key not provided."
        
        # Intializing NewsAPIClient
        newsapi = NewsApiClient(api_key=news_api_key)

        # Get the response based on query
        response = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=max_results)

        # Get the articles from the response
        articles = response.get('articles', [])

        # Initializing empty string
        formatted = ""

        # Function to loop over the articles
        for i, article in enumerate(articles, start=1):
            # Adding the info about the article to Formatted string
            formatted += f"{i}. {article['title']} ({article['source']['name']})\n{article['url']}\n\n"
        
        # Getting the result
        return formatted or "No news articles found."
    except Exception as e:
        return f"Error fetching NewsAPI data: {str(e)}"

# Function for Web Search based on user query
def get_duckduckgo(query):
    try:
        results = duck_tool.run(query)
        return results or "No DuckDuckGo results found."
    except Exception as e:
        return f"Error fetching DuckDuckGo results: {str(e)}"

# Function for Wikipedia Search based on user quesy
def get_wikipedia(query):
    try:
        result = wiki_tool.run(query)
        return result or "No Wikipedia content found."
    except Exception as e:
        return f"Error fetching Wikipedia: {str(e)}"

# Function for Arxiv Search based on user query
def get_arxiv(query, max_results=3):
    try:
        result = arxiv_tool.run({"query": query, "max_results": max_results})
        return result or "No Arxiv papers found."
    except Exception as e:
        return f"Error fetching Arxiv data: {str(e)}"
    
# Text input for user query
query = st.text_input("🔍 Enter your research question:")

# Handle query submission
if query:
    with st.spinner("🔄 Generating answer..."):
        final_answer = None

        # Generating the answer based on choosen mode
        if mode == "Local Articles":
            local_context = get_local_answer(query)  # Already cleaned via extract_answer
            final_answer = local_context

        elif mode == "Internet News":
            news_context = get_news(query)
            prompt = f"Summarize and analyze the following news content about '{query}':\n\n{news_context}"
            final_answer = llm.invoke(prompt).content

        elif mode == "Web Search":
            duck_context = get_duckduckgo(query)
            prompt = f"Summarize and analyze the following web content about '{query}':\n\n{duck_context}"
            final_answer = llm.invoke(prompt).content

        elif mode == "Wikipedia":
            wiki_context = get_wikipedia(query)
            prompt = f"Provide a factual summary based on Wikipedia for '{query}':\n\n{wiki_context}"
            final_answer = llm.invoke(prompt).content

        elif mode == "Arxiv":
            arxiv_context = get_arxiv(query)
            prompt = f"Summarize the following Arxiv research papers about '{query}':\n\n{arxiv_context}"
            final_answer = llm.invoke(prompt).content

        # Combines all sources answers
        elif mode == "Hybrid":
            local_context = get_local_answer(query)  # cleaned
            news_context = get_news(query)
            duck_context = get_duckduckgo(query)
            wiki_context = get_wikipedia(query)
            arxiv_context = get_arxiv(query)

            prompt = f"""
            Combine insights from all sources to answer the question: {query}

            1️⃣ Local Articles: {local_context}
            2️⃣ News: {news_context}
            3️⃣ DuckDuckGo: {duck_context}
            4️⃣ Wikipedia: {wiki_context}
            5️⃣ Arxiv: {arxiv_context}
            """
            final_answer = llm.invoke(prompt).content

        # Providing the final answer
        if final_answer:
            st.subheader("✅ **Answer:**")
            st.write(final_answer)
        else:
            st.warning("⚠️ No answer could be generated. Please try again.")

