import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate  # ADD THIS IF NOT ALREADY IMPORTED
import logging
import re
from typing import List, Dict, Any
import json
# Custom CB Favicon (SVG data URL)


st.set_page_config(
    page_title="CertamenBot",
    page_icon="./assets/cb_logo.png",
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
@st.cache_resource(show_spinner="Loading knowledge base...")
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
            
            # Simplified approach - use semantic retriever with better parameters
            # BM25 requires too much preprocessing for real-time use
            return _vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": config['max_retrievals'],
                    "fetch_k": 20,  # Fetch more, then filter to best
                    "lambda_mult": 0.7  # Balance relevance vs diversity
                }
            )
        else:
            # Standard retrieval methods
            search_type = "similarity" if retrieval_method == "semantic" else "similarity"
            return _vectorstore.as_retriever(
                search_type=search_type,
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
def create_qa_chain_for_tabs(_vectorstore, _embeddings, model_name, temperature, max_tokens, retrieval_method="hybrid"):
    """Create QA chain that can be used across tabs"""
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
        st.error(f"❌ Error creating QA chain: {e}")
        return None



# --- MAIN APP ---
# Custom logo/header with CB styling
st.markdown("""
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        padding: 1rem 0 2rem 0;
        margin-bottom: 1rem;
    ">
        <div style="
            background: linear-gradient(135deg, #005f73, #0a9396, #94d3a2);
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 32px rgba(0,95,115,0.3);
            margin-right: 1rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -20px;
                right: -20px;
                width: 60px;
                height: 60px;
                background: rgba(255,255,255,0.1);
                border-radius: 50%;
            "></div>
            <span style="
                font-size: 2.5rem;
                font-weight: 900;
                color: white;
                font-family: 'Georgia', serif;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                position: relative;
                z-index: 2;
            ">CB</span>
        </div>
        <div>
            <h1 style="
                margin: 0;
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #005f73, #0a9396);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Georgia', serif;
            ">CertamenBot</h1>
            <p style="
                margin: 0;
                color: #666;
                font-size: 1.1rem;
                font-style: italic;
            ">Your Classical Studies AI Assistant</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.title("🏛️ CertamenBot")
st.warning("⚠️ **Disclaimer:** This AI assistant can make mistakes. Always verify important information with authoritative sources, especially for competitive Certamen preparation.")
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
        st.success("✅ Ready!")
    else:
        st.error("❌ Vectorstore failed")
        st.stop()

# Load enhanced vectorstore
vectorstore, embeddings = load_enhanced_vectorstore(config['vectorstore_path'], config['embedding_model'])
if not vectorstore:
    st.stop()

# Create enhanced QA chain - Recreate on parameter changes
qa = create_qa_chain_for_tabs(vectorstore, embeddings, model_choice, temperature, max_tokens, retrieval_method)

if not qa:
    st.error("❌ Failed to create QA chain")
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
            "💬 Ask anything Certamen:",
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
        st.info("👋 Welcome to CertamenBot! Ask anything Certamen-related for comprehensive answers.")

# Replace the tab2 section in your existing code with this enhanced version

with tab2:
    st.title("🔍 Advanced Search & Mythology Mapping")
    
    # Create sub-tabs for different advanced features
    subtab1, subtab2, subtab3 = st.tabs(["🎯 Smart Search", "🕸️ Relationship Map", "📊 Analytics"])
    
    with subtab1:
        st.subheader("🎯 Advanced Search Options")
        
        # Search filters
        col1, col2 = st.columns(2)
        
        with col1:
            search_category = st.selectbox(
                "Search Category",
                ["All", "Mythology", "History", "Language", "Literature", "Geography"]
            )
            
            time_period = st.selectbox(
                "Historical Period",
                ["All Periods", "Archaic (800-480 BCE)", "Classical (480-323 BCE)", 
                 "Hellenistic (323-146 BCE)", "Roman Republic (509-27 BCE)", 
                 "Roman Empire (27 BCE-476 CE)", "Late Antiquity (284-640 CE)"]
            )
        
        with col2:
            source_type = st.selectbox(
                "Source Type",
                ["All Sources", "Primary Sources", "Archaeological", "Literary", "Historical"]
            )
            
            difficulty = st.selectbox(
                "Difficulty Level",
                ["All Levels", "Novice", "Intermediate", "Advanced", "Expert"]
            )
        
        # Advanced query input
        advanced_query = st.text_area(
            "Advanced Query",
            placeholder="Enter your detailed question with specific requirements...",
            height=100
        )
        
        # Search execution
        if st.button("🔍 Execute Advanced Search", use_container_width=True):
            if advanced_query:
                # Build enhanced query with filters
                filter_context = []
                if search_category != "All":
                    filter_context.append(f"Focus on {search_category.lower()}")
                if time_period != "All Periods":
                    filter_context.append(f"From the {time_period}")
                if source_type != "All Sources":
                    filter_context.append(f"Using {source_type.lower()}")
                
                enhanced_query = advanced_query
                if filter_context:
                    enhanced_query += f" ({', '.join(filter_context)})"
                
                # Execute search (use your existing QA chain)
                with st.spinner("🔍 Executing advanced search..."):
                    try:
                        result = qa({"query": enhanced_query})
                        
                        st.success("✅ Search Complete")
                        st.markdown("### 📝 Results")
                        st.write(result["result"])
                        
                        # Enhanced source analysis
                        if result.get("source_documents"):
                            st.markdown("### 📚 Source Analysis")
                            
                            # Group sources by type/period
                            source_analysis = {}
                            for doc in result["source_documents"]:
                                source = doc.metadata.get('source', 'Unknown')
                                if source not in source_analysis:
                                    source_analysis[source] = []
                                source_analysis[source].append(doc.page_content[:200] + "...")
                            
                            for source, contents in source_analysis.items():
                                with st.expander(f"📖 {source} ({len(contents)} references)"):
                                    for i, content in enumerate(contents, 1):
                                        st.markdown(f"**Reference {i}:** {content}")
                    
                    except Exception as e:
                        st.error(f"Search failed: {e}")
    
    with subtab2:
        st.subheader("🕸️ Mythology Relationship Mapping")
        
        # Comprehensive mythology database
        mythology_db = {
            # Olympic Gods - Core Relationships
            "Zeus": {
                "type": "Olympic God",
                "domain": ["Sky", "Thunder", "Justice", "Law"],
                "roman_name": "Jupiter",
                "parents": ["Cronus", "Rhea"],
                "siblings": ["Hestia", "Demeter", "Hera", "Hades", "Poseidon"],
                "spouses": ["Hera", "Metis", "Themis", "Eurynome", "Mnemosyne", "Leto", "Demeter"],
                "children": ["Athena", "Apollo", "Artemis", "Ares", "Hephaestus", "Hebe", "Eileithyia", "Persephone", "Dionysus", "Hermes", "The Muses", "The Fates", "The Graces"],
                "symbols": ["Thunderbolt", "Eagle", "Oak tree", "Bull"],
                "major_myths": ["Titanomachy", "Birth of Athena", "Europa", "Leda", "Ganymede"],
                "epithets": ["Father of Gods", "Cloud-gatherer", "Thunderer"]
            },
            "Hera": {
                "type": "Olympic Goddess",
                "domain": ["Marriage", "Family", "Women", "Childbirth"],
                "roman_name": "Juno",
                "parents": ["Cronus", "Rhea"],
                "siblings": ["Hestia", "Demeter", "Zeus", "Hades", "Poseidon"],
                "spouse": "Zeus",
                "children": ["Ares", "Hephaestus", "Hebe", "Eileithyia"],
                "symbols": ["Peacock", "Cow", "Crown", "Lotus"],
                "major_myths": ["Judgement of Paris", "Io", "Heracles persecution", "Jason and the Argonauts"],
                "epithets": ["Queen of Gods", "White-armed", "Ox-eyed"]
            },
            "Poseidon": {
                "type": "Olympic God",
                "domain": ["Sea", "Earthquakes", "Horses", "Bulls"],
                "roman_name": "Neptune",
                "parents": ["Cronus", "Rhea"],
                "siblings": ["Hestia", "Demeter", "Hera", "Zeus", "Hades"],
                "spouses": ["Amphitrite"],
                "children": ["Triton", "Theseus", "Pegasus", "Chrysaor", "Polyphemus", "Antaeus"],
                "symbols": ["Trident", "Horse", "Dolphin", "Bull"],
                "major_myths": ["Contest for Athens", "Odyssey", "Trojan War", "Medusa"],
                "epithets": ["Earth-shaker", "Horse-tamer", "Lord of the Sea"]
            },
            "Athena": {
                "type": "Olympic Goddess",
                "domain": ["Wisdom", "Warfare", "Crafts", "Justice"],
                "roman_name": "Minerva",
                "parents": ["Zeus", "Metis"],
                "siblings": ["Apollo", "Artemis", "Ares", "Hephaestus", "Dionysus", "Hermes"],
                "symbols": ["Owl", "Olive tree", "Shield", "Spear", "Aegis"],
                "major_myths": ["Birth from Zeus's head", "Contest for Athens", "Arachne", "Perseus", "Odyssey"],
                "epithets": ["Grey-eyed", "Pallas", "City-protector", "Worker"]
            },
            "Apollo": {
                "type": "Olympic God",
                "domain": ["Sun", "Music", "Poetry", "Prophecy", "Healing", "Archery"],
                "roman_name": "Apollo",
                "parents": ["Zeus", "Leto"],
                "siblings": ["Artemis"],
                "children": ["Asclepius", "Orpheus", "Ion", "Aristaeus"],
                "symbols": ["Lyre", "Bow", "Laurel", "Python", "Sun chariot"],
                "major_myths": ["Birth on Delos", "Python slaying", "Daphne", "Cassandra", "Trojan War"],
                "epithets": ["Phoebus", "Far-shooter", "Lycian", "Pythian"]
            },
            "Artemis": {
                "type": "Olympic Goddess",
                "domain": ["Hunt", "Moon", "Chastity", "Childbirth", "Wild animals"],
                "roman_name": "Diana",
                "parents": ["Zeus", "Leto"],
                "siblings": ["Apollo"],
                "symbols": ["Silver bow", "Deer", "Cypress tree", "Amaranth flower"],
                "major_myths": ["Birth on Delos", "Actaeon", "Orion", "Niobe", "Trojan War"],
                "epithets": ["Huntress", "Silver-bowed", "Lady of wild things"]
            },
            "Aphrodite": {
                "type": "Olympic Goddess",
                "domain": ["Love", "Beauty", "Pleasure", "Procreation"],
                "roman_name": "Venus",
                "parents": ["Uranus foam", "Zeus and Dione"],
                "spouse": "Hephaestus",
                "lovers": ["Ares", "Adonis", "Anchises"],
                "children": ["Eros", "Aeneas", "Harmonia", "Phobos", "Deimos"],
                "symbols": ["Dove", "Rose", "Myrtle", "Swan", "Sparrow"],
                "major_myths": ["Birth from sea foam", "Judgement of Paris", "Trojan War", "Adonis", "Pygmalion"],
                "epithets": ["Golden", "Laughter-loving", "Cyprus-born"]
            },
            "Ares": {
                "type": "Olympic God",
                "domain": ["War", "Courage", "Battle fury"],
                "roman_name": "Mars",
                "parents": ["Zeus", "Hera"],
                "lovers": ["Aphrodite"],
                "children": ["Phobos", "Deimos", "Harmonia", "Amazons"],
                "symbols": ["Spear", "Shield", "Helmet", "Vulture", "Dog"],
                "major_myths": ["Affair with Aphrodite", "Trojan War", "Amazons", "Cadmus"],
                "epithets": ["Slayer of men", "Bronze-armoured", "Bane of mortals"]
            },
            "Hephaestus": {
                "type": "Olympic God",
                "domain": ["Fire", "Metalworking", "Crafts", "Sculpture"],
                "roman_name": "Vulcan",
                "parents": ["Zeus", "Hera"],
                "spouse": "Aphrodite",
                "symbols": ["Hammer", "Anvil", "Fire", "Donkey", "Crane"],
                "major_myths": ["Thrown from Olympus", "Pandora", "Achilles' armor", "Prometheus"],
                "epithets": ["Lame god", "Famous craftsman", "Fire-god"]
            },
            "Demeter": {
                "type": "Olympic Goddess",
                "domain": ["Agriculture", "Harvest", "Nature", "Seasons"],
                "roman_name": "Ceres",
                "parents": ["Cronus", "Rhea"],
                "siblings": ["Hestia", "Hera", "Zeus", "Hades", "Poseidon"],
                "children": ["Persephone", "Plutus", "Arion"],
                "symbols": ["Wheat", "Cornucopia", "Torch", "Bread"],
                "major_myths": ["Persephone's abduction", "Eleusinian Mysteries", "Demophon"],
                "epithets": ["Bringer of seasons", "Giver of grain", "Lady of the harvest"]
            },
            "Dionysus": {
                "type": "Olympic God",
                "domain": ["Wine", "Festivity", "Theater", "Madness"],
                "roman_name": "Bacchus",
                "parents": ["Zeus", "Semele"],
                "symbols": ["Grapevine", "Ivy", "Thyrsus", "Leopard", "Goat"],
                "major_myths": ["Birth from Zeus's thigh", "Pentheus", "Pirates", "Ariadne"],
                "epithets": ["Twice-born", "Bull-horned", "Ivy-crowned"]
            },
            "Hermes": {
                "type": "Olympic God",
                "domain": ["Messengers", "Trade", "Thieves", "Travel", "Boundaries"],
                "roman_name": "Mercury",
                "parents": ["Zeus", "Maia"],
                "children": ["Pan", "Hermaphroditus", "Autolycus"],
                "symbols": ["Caduceus", "Winged sandals", "Petasos hat", "Tortoise"],
                "major_myths": ["Birth and cattle theft", "Argus slaying", "Perseus", "Pandora"],
                "epithets": ["Messenger", "Guide of souls", "Giant-killer"]
            },
            
            # Titans
            "Cronus": {
                "type": "Titan",
                "domain": ["Time", "Harvest"],
                "roman_name": "Saturn",
                "parents": ["Uranus", "Gaia"],
                "spouse": "Rhea",
                "children": ["Hestia", "Demeter", "Hera", "Hades", "Poseidon", "Zeus"],
                "major_myths": ["Castration of Uranus", "Devouring children", "Titanomachy"],
                "epithets": ["Father Time", "Harvester"]
            },
            "Rhea": {
                "type": "Titaness",
                "domain": ["Fertility", "Motherhood"],
                "roman_name": "Ops",
                "parents": ["Uranus", "Gaia"],
                "spouse": "Cronus",
                "children": ["Hestia", "Demeter", "Hera", "Hades", "Poseidon", "Zeus"],
                "major_myths": ["Saving Zeus", "Mother of gods"],
                "epithets": ["Mother of gods", "Great mother"]
            },
            
            # Underworld
            "Hades": {
                "type": "Chthonic God",
                "domain": ["Underworld", "Death", "Wealth"],
                "roman_name": "Pluto",
                "parents": ["Cronus", "Rhea"],
                "siblings": ["Hestia", "Demeter", "Hera", "Zeus", "Poseidon"],
                "spouse": "Persephone",
                "symbols": ["Helmet of invisibility", "Cypress", "Narcissus", "Key"],
                "major_myths": ["Persephone's abduction", "Orpheus and Eurydice", "Sisyphus"],
                "epithets": ["Rich one", "Host of many", "Invisible"]
            },
            "Persephone": {
                "type": "Chthonic Goddess",
                "domain": ["Spring", "Underworld", "Death and rebirth"],
                "roman_name": "Proserpina",
                "parents": ["Zeus", "Demeter"],
                "spouse": "Hades",
                "symbols": ["Pomegranate", "Flowers", "Torch"],
                "major_myths": ["Abduction by Hades", "Return to earth", "Eleusinian Mysteries"],
                "epithets": ["Kore", "Iron queen", "Dread queen"]
            }
        }
        
        # Character selection for mapping
        col1, col2 = st.columns(2)
        
        with col1:
            selected_character = st.selectbox(
                "Select Character for Relationship Map",
                list(mythology_db.keys()),
                index=0
            )
        
        with col2:
            map_depth = st.selectbox(
                "Relationship Depth",
                ["Direct (1 level)", "Extended (2 levels)", "Complete Network (3 levels)"],
                index=1
            )
        
        if selected_character:
            character_data = mythology_db[selected_character]
            
            # Display character profile
            st.markdown(f"### 👑 {selected_character}")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Type:** {character_data['type']}")
                st.markdown(f"**Roman Name:** {character_data.get('roman_name', 'N/A')}")
                
            with col2:
                domains = ", ".join(character_data.get('domain', []))
                st.markdown(f"**Domain:** {domains}")
                
            with col3:
                epithets = ", ".join(character_data.get('epithets', []))
                st.markdown(f"**Epithets:** {epithets}")
            
            # Family relationships
            st.markdown("#### 👨‍👩‍👧‍👦 Family Tree")
            
            family_cols = st.columns(4)
            
            with family_cols[0]:
                if 'parents' in character_data:
                    st.markdown("**Parents:**")
                    for parent in character_data['parents']:
                        if parent in mythology_db:
                            st.markdown(f"• [{parent}](#{parent.lower()})")
                        else:
                            st.markdown(f"• {parent}")
            
            with family_cols[1]:
                if 'siblings' in character_data:
                    st.markdown("**Siblings:**")
                    for sibling in character_data['siblings'][:5]:  # Limit display
                        if sibling in mythology_db:
                            st.markdown(f"• [{sibling}](#{sibling.lower()})")
                        else:
                            st.markdown(f"• {sibling}")
                    if len(character_data['siblings']) > 5:
                        st.markdown(f"• ... and {len(character_data['siblings']) - 5} more")
            
            with family_cols[2]:
                spouses = []
                if 'spouse' in character_data:
                    spouses = [character_data['spouse']]
                elif 'spouses' in character_data:
                    spouses = character_data['spouses']
                
                if spouses:
                    st.markdown("**Spouses:**")
                    for spouse in spouses[:3]:
                        if spouse in mythology_db:
                            st.markdown(f"• [{spouse}](#{spouse.lower()})")
                        else:
                            st.markdown(f"• {spouse}")
            
            with family_cols[3]:
                if 'children' in character_data:
                    st.markdown("**Children:**")
                    for child in character_data['children'][:5]:
                        if child in mythology_db:
                            st.markdown(f"• [{child}](#{child.lower()})")
                        else:
                            st.markdown(f"• {child}")
                    if len(character_data['children']) > 5:
                        st.markdown(f"• ... and {len(character_data['children']) - 5} more")
            
            # Symbols and attributes
            st.markdown("#### 🔮 Symbols & Attributes")
            if 'symbols' in character_data:
                symbols_text = " • ".join(character_data['symbols'])
                st.markdown(f"**Sacred Symbols:** {symbols_text}")
            
            # Major myths
            if 'major_myths' in character_data:
                st.markdown("#### 📖 Major Myths & Stories")
                myth_cols = st.columns(2)
                for i, myth in enumerate(character_data['major_myths']):
                    with myth_cols[i % 2]:
                        st.markdown(f"• **{myth}**")
            
            # Interactive relationship network
            st.markdown("#### 🕸️ Relationship Network")
            
            # Create network visualization data
            network_data = {
                "nodes": [{"id": selected_character, "type": character_data['type'], "level": 0}],
                "edges": []
            }
            
            # Add direct relationships
            def add_relationships(char_name, char_data, level, max_level):
                if level >= max_level:
                    return
                
                relationship_types = ['parents', 'siblings', 'children', 'spouse', 'spouses', 'lovers']
                
                for rel_type in relationship_types:
                    if rel_type in char_data:
                        relations = char_data[rel_type]
                        if isinstance(relations, str):
                            relations = [relations]
                        
                        for relation in relations:
                            if relation in mythology_db:
                                # Add node if not exists
                                if not any(node['id'] == relation for node in network_data['nodes']):
                                    network_data['nodes'].append({
                                        "id": relation,
                                        "type": mythology_db[relation]['type'],
                                        "level": level + 1
                                    })
                                
                                # Add edge
                                network_data['edges'].append({
                                    "from": char_name,
                                    "to": relation,
                                    "type": rel_type
                                })
                                
                                # Recursive call for deeper levels
                                if level + 1 < max_level:
                                    add_relationships(relation, mythology_db[relation], level + 1, max_level)
            
            depth_levels = {"Direct (1 level)": 1, "Extended (2 levels)": 2, "Complete Network (3 levels)": 3}
            max_depth = depth_levels[map_depth]
            
            add_relationships(selected_character, character_data, 0, max_depth)
            
            # Display network summary
            st.markdown(f"**Network Summary:** {len(network_data['nodes'])} characters, {len(network_data['edges'])} relationships")
            
            # Group by relationship type
            relationship_summary = {}
            for edge in network_data['edges']:
                rel_type = edge['type']
                if rel_type not in relationship_summary:
                    relationship_summary[rel_type] = []
                relationship_summary[rel_type].append(f"{edge['from']} ↔ {edge['to']}")
            
            # Display relationship types
            for rel_type, relationships in relationship_summary.items():
                with st.expander(f"🔗 {rel_type.title()} ({len(relationships)} connections)"):
                    for rel in relationships[:10]:  # Limit display
                        st.markdown(f"• {rel}")
                    if len(relationships) > 10:
                        st.markdown(f"• ... and {len(relationships) - 10} more")
    
    with subtab3:
        st.subheader("📊 Knowledge Analytics")
        
        # Analytics for the knowledge base
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Database Statistics")
            st.metric("Mythology Characters", len(mythology_db))
            st.metric("Relationship Types", 8)
            st.metric("Cross-references", sum(len(char.get('children', [])) + len(char.get('siblings', [])) for char in mythology_db.values()))
        
        with col2:
            st.markdown("#### 🏛️ Character Types")
            type_counts = {}
            for char_data in mythology_db.values():
                char_type = char_data['type']
                type_counts[char_type] = type_counts.get(char_type, 0) + 1
            
            for char_type, count in type_counts.items():
                st.metric(char_type, count)
        
        # Most connected characters
        st.markdown("#### 🌟 Most Connected Characters")
        connections = {}
        for char_name, char_data in mythology_db.items():
            connection_count = 0
            for rel_type in ['parents', 'siblings', 'children', 'spouse', 'spouses', 'lovers']:
                if rel_type in char_data:
                    if isinstance(char_data[rel_type], list):
                        connection_count += len(char_data[rel_type])
                    else:
                        connection_count += 1
            connections[char_name] = connection_count
        
        # Sort and display top connected
        top_connected = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (char, count) in enumerate(top_connected, 1):
            st.markdown(f"**{i}.** {char} - {count} connections")
        
        # Search patterns
        st.markdown("#### 🔍 Common Search Patterns")
        common_queries = [
            "Who is [character name]?",
            "What are the symbols of [deity]?",
            "Tell me about [myth name]",
            "Who are the children of [parent]?",
            "What is the Roman name for [Greek god]?",
            "Explain the myth of [story]"
        ]
        
        for query in common_queries:
            st.markdown(f"• {query}")

with tab3:
    st.title("🏛️ About CertamenBot")
    
    st.markdown("""
    ### Enhanced Features
    
    CertamenBot includes several improvements for better name and entity retrieval:
    
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








