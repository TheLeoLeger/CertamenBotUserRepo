
import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- ENVIRONMENT VARIABLES ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")  # default if not set

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is not set. The ChatOpenAI model will not work.")
st.write("✅ Loaded OpenAI key?", bool(OPENAI_API_KEY))
st.write("🔑 Key starts with:", OPENAI_API_KEY[:5] + "..." if OPENAI_API_KEY else "None")


# --- AUTO-DOWNLOAD VECTORSTORE FILES ---
def download_vectorstore():
    import gdown
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    files = {
        "index.faiss": "https://drive.google.com/uc?id=1upeOyb4VLWdP1cE-mIIDSuWtCyrP7SEw",
        "index.pkl": "https://drive.google.com/uc?id=1yjKByVSebOPoZkYR-0EmxA5uwfM32FIE"
    }

    for filename, url in files.items():
        filepath = os.path.join(VECTORSTORE_PATH, filename)
        if not os.path.exists(filepath):
            with st.spinner(f"📥 Downloading {filename}..."):
                gdown.download(url, filepath, quiet=False)

download_vectorstore()

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="📚 PDF AI Chat", layout="wide")
st.markdown("""
    <style>
    body { background-color: #111; color: #eee; }
    .chat-user { background: #005f73; color: white; padding:8px; border-radius:8px; margin:4px 0; text-align:right; }
    .chat-ai { background: #0a9396; color:white; padding:8px; border-radius:8px; margin:4px 0; text-align:left; }
    input, button { background: #222; color: white !important; }
    </style>
""", unsafe_allow_html=True)
st.title("📚 AI Chat from Your PDFs (OCR‑enabled)")

# --- LOAD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

vectorstore = load_vectorstore(VECTORSTORE_PATH)
if not vectorstore:
    st.stop()

# --- INIT QA CHAIN ---
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# --- USER QUERY ---
query = st.text_input("Ask something about your sourcebooks:", key="user_input")
if query:
    with st.spinner("Thinking..."):
        res = qa({"query": query})
        ans = res["result"]
        srcs = res.get("source_documents", [])
        st.session_state.history.append((query, ans, srcs))

# --- SHOW CHAT HISTORY ---
for q, ans, sources in st.session_state.history:
    st.markdown(f"<div class='chat-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'><b>AI:</b> {ans}</div>", unsafe_allow_html=True)
    if sources:
        bullets = "\n".join(f"- {doc.metadata.get('source')}" for doc in sources)
        st.markdown(f"**Sources:**\n{bullets}")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
