import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import logging
import re
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CertamenBot Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CONFIGURATION ---
@st.cache_data
def get_config():
    """Get configuration with validation"""
    api_key = None
    
    # Try multiple ways to get the API key
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    config = {
        'openai_api_key': api_key,
        'vectorstore_path': os.getenv("VECTORSTORE_PATH", "vectorstore"),
        'model_name': os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        'max_retrievals': 12,  # Increased for better coverage
        'chunk_overlap': 50,   # For better context
    }
    return config

config = get_config()

# Validate API key
if not config['openai_api_key']:
    st.error("❌ OPENAI_API_KEY is not set in Streamlit secrets.")
    st.stop()

# --- ENHANCED QUERY PROCESSING ---
class EnhancedQueryProcessor:
    """Enhanced query processing for better name/entity matching"""
    
    def __init__(self):
        # Common Certamen name variations and aliases
        self.name_aliases = {
            # Greek to Roman
            'zeus': ['jupiter', 'jove', 'iuppiter'],
            'hera': ['juno'],
            'poseidon': ['neptune', 'neptunus'],
            'athena': ['minerva'],
            'apollo': ['phoebus', 'phoebus apollo'],
            'artemis': ['diana'],
            'aphrodite': ['venus'],
            'ares': ['mars'],
            'hephaestus': ['vulcan', 'vulcanus'],
            'demeter': ['ceres'],
            'dionysus': ['bacchus', 'liber'],
            'hermes': ['mercury', 'mercurius'],
            'hades': ['pluto', 'pluton', 'dis'],
            'persephone': ['proserpina', 'proserpine'],
            'hestia': ['vesta'],
            # Add more as needed
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with aliases and variations"""
        queries = [query]
        query_lower = query.lower()
        
        # Check for name aliases
        for main_name, aliases in self.name_aliases.items():
            if main_name in query_lower:
                for alias in aliases:
                    queries.append(query.replace(main_name, alias, 1))
            for alias in aliases:
                if alias in query_lower:
                    queries.append(query.replace(alias, main_name, 1))
        
        # Add variations (remove articles, plurals, etc.)
        variations = []
        for q in queries:
            # Remove common articles
            clean_q = re.sub(r'\b(the|a|an)\b', '', q, flags=re.IGNORECASE).strip()
            if clean_q and clean_q not in queries:
                variations.append(clean_q)
        
        queries.extend(variations)
        return list(set(queries))  # Remove duplicates
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract potential names/entities from query"""
        # Simple capitalized word extraction (can be enhanced with NER)
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        return entities

# --- AUTO-DOWNLOAD VECTORSTORE FILES ---
@st.cache_data
def download_vectorstore():
    """Download vectorstore files with better error handling"""
    try:
        import gdown
        os.makedirs(config['vectorstore_path'], exist_ok=True)
        
        files = {
            "index.faiss": "https://drive.google.com/uc?id=1ZXBTEg-upb1I_oJbRfr80rVWcy6sKM8L",
            "index.pkl": "https://drive.google.com/uc?id=1JSPsxyqgpe7YbMq_AWQ6KnIQ6eNAX8yj"
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

# --- ENHANCED STYLING ---
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .chat-container { max-height: 500px; overflow-y: auto; }
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
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 0.9em;
        border-left: 4px solid #0a9396;
    }
    .confidence-score {
        background: #e8f4f8;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        color: #005f73;
        font-weight: bold;
    }
    .entity-highlight {
        background: #ffe066;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .quick-action {
        background: #0a9396;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        cursor: pointer;
        font-size: 0.9em;
    }
    .quick-action:hover {
        background: #005f73;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD VECTORSTORE WITH ENHANCED RETRIEVAL ---
@st.cache_resource(show_spinner="Loading enhanced knowledge base...")
def load_enhanced_vectorstore(path, embedding_model):
    """Load vectorstore with enhanced retrieval capabilities"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = FAISS.load_local(
            path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded vectorstore with {vectorstore.index.ntotal} documents")
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {e}")
        logger.error(f"Vectorstore loading failed: {e}")
        return None, None

# --- ENHANCED RETRIEVER ---
@st.cache_resource
def create_enhanced_retriever(_vectorstore, _embeddings, retrieval_method="hybrid"):
    """Create enhanced retriever with multiple search strategies"""
    try:
        if retrieval_method == "hybrid":
            # Create semantic retriever
            semantic_retriever = _vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
            
            # Get all documents for BM25 (this is a simplified approach)
            # In production, you'd want to store documents separately
            docs = []
            try:
                # This is a workaround - in real implementation, store docs separately
                all_docs = _vectorstore.similarity_search("", k=1000)  # Get sample docs
                docs = all_docs
            except:
                docs = []
            
            if docs:
                # Create BM25 retriever for keyword matching
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 8
                
                # Combine retrievers
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.6, 0.4]  # Favor semantic but include keyword
                )
                return ensemble_retriever
            else:
                return semantic_retriever
        else:
            # Fallback to semantic only
            return _vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config['max_retrievals']}
            )
            
    except Exception as e:
        logger.error(f"Enhanced retriever creation failed: {e}")
        # Fallback to basic retriever
        return _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config['max_retrievals']}
        )

# --- ENHANCED QA CHAIN ---
@st.cache_resource
def create_enhanced_qa_chain(_vectorstore, _embeddings, model_name, temperature, max_tokens, retrieval_method="hybrid"):
    """Create enhanced QA chain with better retrieval"""
    try:
        llm = ChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=config['openai_api_key'],
            model_name=model_name
        )
        
        # Create enhanced retriever
        retriever = create_enhanced_retriever(_vectorstore, _embeddings, retrieval_method)
        
        # Enhanced prompt template
        from langchain.prompts import PromptTemplate
        
        prompt_template = """
        You are CertamenBot, an expert in Classical studies, mythology, history, and Latin/Greek languages.
        
        Use the following pieces of context to answer the question. When answering:
        1. Be specific and cite sources when possible
        2. If you find specific names or entities, provide comprehensive information
        3. Include relevant cross-references (e.g., Greek/Roman equivalents)
        4. If the question is about a specific person/character, provide their key attributes, stories, and significance
        5. If you're not completely certain, say so
        
        Context: {context}
        
        Question: {question}
        
        Detailed Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa
    except Exception as e:
        st.error(f"❌ Error creating enhanced QA chain: {e}")
        return None

# --- MAIN APP ---
st.title("🏛️ CertamenBot Pro")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🏛️ Chat", "🔍 Advanced Search", "ℹ️ About"])

# Initialize query processor
query_processor = EnhancedQueryProcessor()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Retrieval method
    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["hybrid", "semantic", "keyword"],
        index=0,
        help="Hybrid combines semantic understanding with keyword matching for better name/entity retrieval"
    )
    
    # Temperature slider
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.1, 0.1)
    
    # Max tokens
    max_tokens = st.number_input("Max Response Length", 100, 4000, 1500)
    
    # Enhanced search options
    st.subheader("🔍 Search Options")
    use_query_expansion = st.checkbox("Expand Queries", value=True, help="Automatically include name aliases and variations")
    show_confidence = st.checkbox("Show Confidence", value=True, help="Display confidence scores for answers")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    # Show vectorstore status
    st.subheader("📊 Status")
    if download_vectorstore():
        st.success("✅ Enhanced system ready!")
    else:
        st.error("❌ Vectorstore failed")
        st.stop()

# Load enhanced vectorstore
vectorstore, embeddings = load_enhanced_vectorstore(config['vectorstore_path'], config['embedding_model'])
if not vectorstore:
    st.stop()

# Create enhanced QA chain
qa = create_enhanced_qa_chain(vectorstore, embeddings, model_choice, temperature, max_tokens, retrieval_method)
if not qa:
    st.stop()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

with tab1:
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⚡ Random Myth"):
            st.session_state.quick_query = "Tell me about a famous Greek or Roman myth"
    with col2:
        if st.button("👑 Gods & Goddesses"):
            st.session_state.quick_query = "List the major Olympian gods and their roles"
    with col3:
        if st.button("🏛️ Roman History"):
            st.session_state.quick_query = "Tell me about important Roman historical events"
    with col4:
        if st.button("📚 Etymology"):
            st.session_state.quick_query = "Explain the etymology of a Latin word"
    
    # Main input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "💬 Ask about any name, place, event, or concept in Classical studies:",
            key="user_input",
            placeholder="e.g., Who is Minerva? What happened at the Battle of Actium? Etymology of 'democracy'",
            value=st.session_state.get("quick_query", "")
        )
        
        if "quick_query" in st.session_state:
            del st.session_state.quick_query
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Show expanded queries if enabled
    if query and use_query_expansion:
        expanded_queries = query_processor.expand_query(query)
        if len(expanded_queries) > 1:
            with st.expander("🔄 Query Variations", expanded=False):
                for i, eq in enumerate(expanded_queries[:5]):  # Show first 5
                    st.text(f"{i+1}. {eq}")
    
    # Process query
    if query and (search_button or query != st.session_state.get("last_query", "")):
        st.session_state.last_query = query
        
        with st.spinner("🤔 Searching through ancient texts..."):
            try:
                # Use expanded queries if enabled
                queries_to_try = query_processor.expand_query(query) if use_query_expansion else [query]
                
                best_result = None
                best_score = 0
                
                # Try each query variation
                for q in queries_to_try[:3]:  # Limit to first 3 to avoid too many API calls
                    try:
                        result = qa({"query": q})
                        # Simple scoring based on answer length and source count
                        score = len(result["result"]) + len(result.get("source_documents", [])) * 50
                        if score > best_score:
                            best_result = result
                            best_score = score
                    except:
                        continue
                
                if best_result:
                    answer = best_result["result"]
                    sources = best_result.get("source_documents", [])
                    
                    # Extract entities from original query
                    entities = query_processor.extract_entities(query)
                    
                    # Add to history
                    st.session_state.history.append({
                        "query": query,
                        "answer": answer,
                        "sources": sources,
                        "entities": entities,
                        "confidence": min(100, best_score // 10),  # Simple confidence calculation
                        "timestamp": st.session_state.get("timestamp", 0) + 1
                    })
                    st.session_state.timestamp = st.session_state.get("timestamp", 0) + 1
                else:
                    st.error("❌ No results found. Try rephrasing your question.")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                logger.error(f"Query processing failed: {e}")
    
    # Display chat history
    if st.session_state.history:
        st.subheader("💬 Chat History")
        
        for item in reversed(st.session_state.history):
            query_text = item["query"]
            answer_text = item["answer"]
            sources = item["sources"]
            entities = item.get("entities", [])
            confidence = item.get("confidence", 0)
            
            # User message with entity highlighting
            highlighted_query = query_text
            for entity in entities:
                highlighted_query = highlighted_query.replace(
                    entity, 
                    f"<span class='entity-highlight'>{entity}</span>"
                )
            
            st.markdown(
                f"<div class='chat-user'><b>You:</b> {highlighted_query}</div>", 
                unsafe_allow_html=True
            )
            
            # AI response with confidence
            confidence_badge = ""
            if show_confidence and confidence > 0:
                confidence_badge = f"<span class='confidence-score'>Confidence: {confidence}%</span>"
            
            st.markdown(
                f"<div class='chat-ai'><b>CertamenBot:</b> {answer_text} {confidence_badge}</div>", 
                unsafe_allow_html=True
            )
            
            # Enhanced sources display
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        source_name = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        
                        # Highlight entities in content preview
                        for entity in entities:
                            content_preview = re.sub(
                                f"\\b{re.escape(entity)}\\b",
                                f"**{entity}**",
                                content_preview,
                                flags=re.IGNORECASE
                            )
                        
                        st.markdown(f"""
                        <div class='source-item'>
                        <strong>Source {i}:</strong> {source_name} (Page: {page})<br>
                        <em>Content:</em> {content_preview}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
    
    else:
        st.info("👋 Welcome to CertamenBot Pro! Ask about any Classical name, place, or concept for comprehensive answers.")

with tab2:
    st.title("🔍 Advanced Search")
    st.markdown("Coming soon: Advanced filtering, timeline view, and relationship mapping!")
    
    # Placeholder for advanced features
    st.info("🚧 Advanced features in development:\n- Filter by source type\n- Timeline visualization\n- Character relationship maps\n- Etymology trees")

with tab3:
    st.title("🏛️ About CertamenBot Pro")
    
    st.markdown("""
    ### Enhanced Features
    
    CertamenBot Pro includes several improvements for better name and entity retrieval:
    
    **🔍 Enhanced Search:**
    - **Hybrid Retrieval**: Combines semantic understanding with keyword matching
    - **Query Expansion**: Automatically includes Greek/Roman name variants
    - **Entity Recognition**: Better handling of proper names and places
    
    **🎯 Smart Features:**
    - **Confidence Scoring**: Shows how confident the AI is in its answers
    - **Source Highlighting**: Highlights found entities in source documents
    - **Quick Actions**: Common query templates for faster searching
    
    **📚 Better Coverage:**
    - Increased retrieval count for comprehensive answers
    - Multiple search strategies to find specific names
    - Enhanced prompting for more detailed responses
    
    ### Developer
    **Main Developer:** Leo Leger  
    **Contact:** [leoallanleger@gmail.com](mailto:leoallanleger@gmail.com)
    
    ### Resources
    **Sourcebooks:** [NJCL Sourcebook Collection](https://drive.google.com/drive/u/1/folders/12GCk3D9KQksf1iBC2jgfLU91pZTnDeS6?dmr=1&ec=wgc-drive-globalnav-goto)
    
    ---
    
    *Built with ❤️ for the Certamen community*
    """)
    
    # Enhanced stats
    st.subheader("📊 System Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 AI Model", model_choice)
    
    with col2:
        if vectorstore:
            st.metric("📚 Documents", f"{vectorstore.index.ntotal:,}")
        else:
            st.metric("📚 Documents", "Loading...")
    
    with col3:
        st.metric("🔍 Retrieval", retrieval_method.title())
    
    with col4:
        st.metric("📈 Max Results", config['max_retrievals'])

# Footer
st.markdown("---")
st.markdown("*Enhanced with hybrid search, query expansion, and intelligent entity recognition*")
