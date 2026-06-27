import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. INSTANT PAGE SETUP ---
st.set_page_config(
    page_title="TN Citizen Scheme AI Portal", 
    page_icon="🏛️", 
    layout="wide"
)

# Custom Enterprise-Grade Styling
st.markdown("""
    <style>
        /* Base page styling */
        .main { background-color: #f8fafc; }
        
        /* Premium Header Banner */
        .header-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.2);
            border-left: 8px solid #eab308;
        }
        .header-banner h1 { color: #ffffff !important; margin: 0; font-weight: 700; font-size: 2.2rem; }
        .header-banner p { color: #93c5fd !important; font-size: 1.1rem; margin-top: 0.5rem; }
        
        /* Professional Side Navigation Overhaul */
        [data-testid="stSidebar"] {
            background-color: #0b0f19 !important;
            border-right: 1px solid #1e293b;
        }
        
        /* Styling text inside side navigation */
        [data-testid="stSidebar"] h2 {
            color: #ffffff !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            margin-bottom: 0px !important;
        }
        
        [data-testid="stSidebar"] h3 {
            color: #94a3b8 !important;
            font-size: 0.95rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #cbd5e1 !important;
        }
        
        /* Custom layout for info and success blocks inside the side nav */
        [data-testid="stSidebar"] .stAlert {
            background-color: #111827 !important;
            border: 1px solid #1e293b !important;
            color: #e2e8f0 !important;
            border-radius: 8px !important;
        }
        
        /* Chat bubble adjustments */
        .stChatMessage { border-radius: 8px; margin-bottom: 0.75rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR METADATA CONTAINER ---
with st.sidebar:
    st.markdown("## 🏛️ Engine Dashboard")
    st.markdown("---")
    st.markdown("### 📊 Status Metrics")
    
    if os.path.exists("./chroma_db"):
        st.success("Vector DB: Persistent Storage Connected ✅")
    else:
        st.error("Vector DB: Storage Missing! Run ingest.py ❌")
        st.stop()
        
    st.markdown("### ⚙️ Buildathon Stack")
    st.caption("LLM Base: `gpt-4o-mini`")
    st.caption("Vector Core: `ChromaDB` (Local)")
    st.caption("Embeddings: `all-MiniLM-L6-v2`")
    st.caption("Framework: `LangChain (LCEL)`")
    st.markdown("---")
    st.info("💡 *This application matches text structures semantic-by-semantic to cross-verify citizenship policy benefits.*")

# Main Banner Layout
st.markdown("""
    <div class="header-banner">
        <h1>Tamil Nadu Welfare Schemes AI Portal</h1>
        <p>Enterprise-grade semantic retrieval assistant serving validated policy and subsidy guidelines.</p>
    </div>
""", unsafe_allow_html=True)

# --- 3. LIFECYCLE INITIALIZATION ---
@st.cache_resource
def load_inference_chain():
    """Loads local vector store and connects OpenAI chain instantly."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    template = """You are an official enterprise assistant helping citizens discover Tamil Nadu Government Welfare Schemes.
    Use the following pieces of retrieved document context to answer the question concisely and professionally.
    
    Rule: If the exact answer or specific program cannot be cross-referenced from the provided context matrices, say:
    "I cannot find that specific information in the current index. Please try refining your query or check back later."
    Do not invent facts.

    Context Matrix:
    {context}

    Citizen Question: {question}
    Verified System Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if not os.environ.get("OPENAI_API_KEY"):
    st.error("❌ System Error: OPENAI_API_KEY environment token missing.")
    st.stop()

rag_chain = load_inference_chain()

# --- 4. CONVERSATIONAL LOOP CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Vanakkam! I am your AI Schemes Assistant. Ask me any policy question regarding subsidies or training programs."}
    ]

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt interaction
if user_query := st.chat_input("Enter your query (e.g., 'What benefits are available for MSMEs?')..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing semantic query..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})