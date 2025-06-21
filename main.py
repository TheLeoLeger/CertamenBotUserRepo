import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG (Move to top) ---
st.set_page_config(
    page_title="📚 PDF AI Chat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENVIRONMENT VARIABLES ---
@st.cache_data
def get_config():
    """Get configuration with validation"""
    config = {
        'openai_api_key': st.secrets.get("OPENAI_API_KEY"),
        'vectorstore_path': os.getenv("VECTORSTORE_PATH", "vectorstore"),
        'model_name': os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    }
    return config

config = get_config()

# Validate API key
if not config['openai_api_key']:
    st.error("❌ OPENAI_API_KEY is not set in Streamlit secrets.")
    st.stop()

# --- AUTO-DOWNLOAD VECTORSTORE FILES ---
@st.cache_data
def download_vectorstore():
    """Download vectorstore files with better error handling"""
    try:
        import gdown
        os.makedirs(config['vectorstore_path'], exist_ok=True)
        
        files = {
            "index.faiss": "https://drive.google.com/uc?id=1upeOyb4VLWdP1cE-mIIDSuWtCyrP7SEw",
            "index.pkl": "https://drive.google.com/uc?id=1yjKByVSebOPoZkYR-0EmxA5uwfM32FIE"
        }
        
        for filename, url in files.items():
            filepath = os.path.join(config['vectorstore_path'], filename)
            if not os.path.exists(filepath):
                with st.spinner(f"📥 Downloading {filename}..."):
                    try:
                        gdown.download(url, filepath, quiet=False)
                        logger.info(f"Downloaded {filename}")
                    except Exception as e:
                        st.error(f"Failed to download {filename}: {e}")
                        return False
        return True
    except ImportError:
        st.error("gdown package not installed. Please install with: pip install gdown")
        return False
    except Exception as e:
        st.error(f"Error setting up vectorstore: {e}")
        return False

# --- STYLING ---
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .chat-container { max-height: 400px; overflow-y: auto; }
    .chat-user { 
        background: linear-gradient(135deg, #005f73, #0a9396); 
        color: white; 
        padding: 12px; 
        border-radius: 12px; 
        margin: 8px 0; 
        text-align: right;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-ai { 
        background: linear-gradient(135deg, #0a9396, #94d3a2); 
        color: white; 
        padding: 12px; 
        border-radius: 12px; 
        margin: 8px 0; 
        text-align: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-item {
        background: #f0f2f6;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px 0;
        font-size: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAIN APP ---
st.title("📚 AI Chat from Your PDFs (OCR‑enabled)")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.1, 0.1)
    
    # Max tokens
    max_tokens = st.number_input("Max Response Length", 100, 4000, 1000)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    # Show vectorstore status
    st.subheader("📊 Status")
    if download_vectorstore():
        st.success("✅ Vectorstore ready")
    else:
        st.error("❌ Vectorstore failed")
        st.stop()

# --- LOAD VECTORSTORE ---
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore(path, embedding_model):
    """Load vectorstore with proper error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = FAISS.load_local(
            path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded vectorstore with {vectorstore.index.ntotal} documents")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {e}")
        logger.error(f"Vectorstore loading failed: {e}")
        return None

vectorstore = load_vectorstore(config['vectorstore_path'], config['embedding_model'])
if not vectorstore:
    st.stop()

# --- INIT QA CHAIN ---
@st.cache_resource
def create_qa_chain(_vectorstore, model_name, temperature, max_tokens):
    """Create QA chain with specified parameters"""
    try:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=config['openai_api_key']
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Return top 4 most relevant chunks
            ),
            return_source_documents=True,
        )
        return qa
    except Exception as e:
        st.error(f"❌ Error creating QA chain: {e}")
        return None

qa = create_qa_chain(vectorstore, model_choice, temperature, max_tokens)
if not qa:
    st.stop()

# --- INITIALIZE SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- MAIN INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "💬 Ask something about your documents:",
        key="user_input",
        placeholder="e.g., What are the main topics covered?"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# --- PROCESS QUERY ---
if query and (search_button or query != st.session_state.get("last_query", "")):
    st.session_state.last_query = query
    
    with st.spinner("🤔 Thinking..."):
        try:
            result = qa({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Add to history
            st.session_state.history.append({
                "query": query,
                "answer": answer,
                "sources": sources,
                "timestamp": st.session_state.get("timestamp", 0) + 1
            })
            st.session_state.timestamp = st.session_state.get("timestamp", 0) + 1
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            logger.error(f"Query processing failed: {e}")

# --- DISPLAY CHAT HISTORY ---
if st.session_state.history:
    st.subheader("💬 Chat History")
    
    # Reverse order to show newest first
    for item in reversed(st.session_state.history):
        query_text = item["query"]
        answer_text = item["answer"]
        sources = item["sources"]
        
        # User message
        st.markdown(
            f"<div class='chat-user'><b>You:</b> {query_text}</div>", 
            unsafe_allow_html=True
        )
        
        # AI response
        st.markdown(
            f"<div class='chat-ai'><b>AI:</b> {answer_text}</div>", 
            unsafe_allow_html=True
        )
        
        # Sources (if available)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    
                    st.markdown(f"""
                    **Source {i}:** {source_name} (Page: {page})
                    
                    *Preview:* {content_preview}
                    """)
        
        st.divider()

else:
    st.info("👋 Welcome! Ask a question about your documents to get started.")

# --- FOOTER ---
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, and OpenAI*")
