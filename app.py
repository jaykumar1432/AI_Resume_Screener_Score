"""Two-page resume screener: Home indexes PDFs, Analyze searches saved index."""
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from pipeline import (
    EMBEDDING_MODEL, LLM_MODEL, load_pdf, chunk_documents,
    create_embeddings, build_faiss, rank_resumes, connect_llm, explain_candidate,
)

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# Custom CSS for Professional Dashboard Look
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #4B5563;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

INDEX_DIR = Path(__file__).resolve().parent / "stored_faiss"
MANIFEST = INDEX_DIR / "manifest.json"

@st.cache_resource(show_spinner="Loading the embedding model...")
def embeddings_model():
    return create_embeddings()

def save_index(store, names):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    MANIFEST.write_text(json.dumps({"resumes": names, "embedding_model": EMBEDDING_MODEL}, indent=2))

def load_index():
    if not MANIFEST.exists() or not (INDEX_DIR / "index.faiss").exists():
        return None, []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("The saved FAISS index uses a different embedding model.")
    index = FAISS.load_local(str(INDEX_DIR), embeddings_model(), allow_dangerous_deserialization=True)
    return index, manifest.get("resumes", [])

st.sidebar.title("📄 AI Resume Screener")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Analyze Resumes", "ℹ️ About"])
st.sidebar.caption("LangChain · HuggingFace · FAISS · RAG · Ollama")

if page == "🏠 Home":
    st.markdown('<p class="main-header">🏠 Home – Upload and Store Resumes</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Upload candidate PDF resumes. The system extracts text, splits chunks, computes local embeddings, and stores them in FAISS.</p>', unsafe_allow_html=True)
    
    uploads = st.file_uploader("Choose PDF resumes", type="pdf", accept_multiple_files=True)
    if st.button("📤 Upload & Store in FAISS", type="primary"):
        if not uploads:
            st.warning("Upload at least one resume PDF.")
        else:
            docs, names = [], []
            for file in uploads:
                if file.name in names:
                    st.warning(f"Skipped duplicate filename: {file.name}")
                    continue
                try:
                    pages = load_pdf(file)
                    if pages:
                        docs.extend(pages)
                        names.append(file.name)
                    else:
                        st.warning(f"No extractable text in {file.name}; skipped.")
                except Exception as err:
                    st.error(f"Could not read {file.name}: {err}")
            
            if docs:
                try:
                    with st.spinner("Chunking, embedding, and storing resumes..."):
                        chunks = chunk_documents(docs)
                        store = build_faiss(chunks, embeddings_model())
                        save_index(store, names)
                    st.success(f"Saved {len(names)} resume(s) as {len(chunks)} chunks.")
                    st.info("Now choose 'Analyze Resumes' from the left menu.")
                except Exception as err:
                    st.error(f"Indexing failed: {err}")
            else:
                st.error("No resume text was extracted.")

    if MANIFEST.exists():
        try:
            saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
            st.subheader("📁 Currently Stored Resumes:")
            for name in saved.get("resumes", []):
                st.write("✅", name)
            st.caption("The index is saved locally in the stored_faiss folder.")
        except Exception:
            st.warning("Stored index metadata could not be read.")

    if st.button("🗑️ Clear stored resumes"):
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
            st.success("Stored index cleared.")
            st.rerun()

elif page == "🔍 Analyze Resumes":
    st.markdown('<p class="main-header">🔍 Analyze Resumes – Enter Job Description</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">This page retrieves resumes already stored on Home. It does not re-upload files.</p>', unsafe_allow_html=True)
    
    if not MANIFEST.exists():
        st.warning("No resumes indexed yet. Go to Home → Upload & Store in FAISS.")
    else:
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
        total_resumes = len(saved.get('resumes', []))
        st.success(f"FAISS is ready: {total_resumes} stored resume(s) available.")
        
        jd = st.text_area("Paste the Job Description (JD)", height=160, placeholder="Paste job description requirements here...")
        
        min_score = st.slider("Minimum Match Threshold (%)", min_value=0, max_value=100, value=70, step=5)
        
        if st.button("🚀 Analyze Resumes", type="primary"):
            if not jd.strip():
                st.warning("Please provide a job description.")
            else:
                try:
                    index, candidate_names = load_index()
                    if not index:
                        st.error("Index could not be loaded.")
                    else:
                        with st.spinner("Retrieving passages and ranking candidates..."):
                            ranked_results = rank_resumes(index, jd, candidate_names, min_match_percentage=min_score)
                            llm = connect_llm()
                        
                        if not ranked_results:
                            st.warning(f"⚠️ No candidates met the minimum match threshold of {min_score}%. Try lowering the slider or review your job description.")
                        else:
                            st.markdown("---")
                            
                            # --- 1. Ranked Candidates Table View with Contact Info ---
                            st.header("📊 Ranked Candidates")
                            
                            table_data = []
                            for rank, item in enumerate(ranked_results, start=1):
                                table_data.append({
                                    "Rank": rank,
                                    "Resume": item["candidate"],
                                    "Email": item["email"],
                                    "Phone": item["phone"],
                                    "Estimated Match Score (%)": f"{int(item['match_percentage'])}%",
                                    "FAISS L2 Distance": round(item["distance"], 4)
                                })
                            
                            df = pd.DataFrame(table_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            st.caption(f"Showing only candidates meeting or exceeding the {min_score}% match threshold with direct contact details.")
                            
                            # --- 2. RAG + Ollama Candidate Explanations with Contacts ---
                            st.header("🤖 RAG + Ollama Candidate Explanations")
                            
                            for rank, item in enumerate(ranked_results, start=1):
                                score_int = int(item['match_percentage'])
                                with st.expander(f"{rank}. {item['candidate']} — {score_int}% Match | 📧 {item['email']} | 📞 {item['phone']}"):
                                    with st.spinner(f"Generating LLM explanation for {item['candidate']}..."):
                                        analysis = explain_candidate(llm, jd, item)
                                        st.markdown(analysis)

                except Exception as err:
                    st.error(f"Analysis failed: {err}")

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About AI Resume Screener</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Architecture
    This application is an **AI-powered Resume Screening System** designed using Retrieval-Augmented Generation (RAG).

    * **Document Processing:** LangChain & PyPDFLoader with automated Regex Contact Extraction (Email & Phone).
    * **Text Chunking:** `RecursiveCharacterTextSplitter` into 650 character chunks with 100 overlap.
    * **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
    * **Vector Search & Scoring:** FAISS vector store with threshold filtering and integer percentage mapping.
    * **RAG & Synthesis:** Ollama local LLM running Qwen models for grounded resume matching explanations.
    """)