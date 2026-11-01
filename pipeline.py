"""PPT flow: LangChain PDF -> cleaning -> chunks -> embeddings -> FAISS -> RAG/Ollama."""
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"

def clean_text(text: str) -> str:
    """Reduce whitespace but preserve punctuation, technical skills and newlines."""
    text = text.replace("\x00", "").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)

def extract_contact_info(text: str):
    """Extract email and phone number using safe regex patterns."""
    # Email regex pattern
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0) if email_match else "Not Found"

    # Phone number regex pattern (handles international, spaces, dashes, parentheses)
    phone_match = re.search(r"\(?\+?\d{1,4}\)?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", text)
    phone = phone_match.group(0) if phone_match else "Not Found"

    return email, phone

def load_pdf(uploaded_file):
    """Load PDF pages with LangChain. PyPDFLoader needs a filesystem path."""
    original_name = Path(uploaded_file.name).name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
        tmp_path = handle.name
        handle.write(uploaded_file.getvalue())
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)
    
    full_text = ""
    for page in pages:
        cleaned = clean_text(page.page_content)
        full_text += "\n" + cleaned
        page.page_content = cleaned
        page.metadata = {"candidate": original_name, "page": page.metadata.get("page", 0)}
    
    # Extract contact details from full document text
    email, phone = extract_contact_info(full_text)
    
    # Attach contact details to the first page's metadata so we can retrieve it easily
    if pages:
        pages[0].metadata["email"] = email
        pages[0].metadata["phone"] = phone

    return [page for page in pages if page.page_content.strip()]

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_faiss(chunks, embeddings):
    return FAISS.from_documents(chunks, embedding=embeddings)

def rank_resumes(store, job_description, names, chunks_per_candidate=3, min_match_percentage=70.0):
    """FAISS retrieves top passages PER candidate, converts L2 distance to an estimated match %,
    and filters out candidates below the min_match_percentage threshold. Extracts contact info.
    """
    ranked = []
    for name in names:
        matches = store.similarity_search_with_score(
            job_description, k=chunks_per_candidate,
            fetch_k=store.index.ntotal, filter={"candidate": name}
        )
        if not matches:
            continue
        
        # Extract email & phone stored in document metadata (isolated from scoring)
        candidate_email = "Not Found"
        candidate_phone = "Not Found"
        for doc, _dist in matches:
            if "email" in doc.metadata and doc.metadata["email"] != "Not Found":
                candidate_email = doc.metadata["email"]
            if "phone" in doc.metadata and doc.metadata["phone"] != "Not Found":
                candidate_phone = doc.metadata["phone"]

        # Default LangChain FAISS score is L2 distance (LOWER is better).
        best_distance = min(distance for _doc, distance in matches)
        
        # Convert L2 distance to integer match percentage
        match_percentage = int(max(0.0, min(100.0, (1.5 - best_distance) * 100)))
        
        # Filter out candidates who fall below the required threshold (e.g., 70%)
        if match_percentage >= min_match_percentage:
            ranked.append({
                "candidate": name,
                "email": candidate_email,
                "phone": candidate_phone,
                "distance": best_distance,
                "match_percentage": match_percentage,
                "passages": [doc.page_content for doc, _dist in matches],
            })
            
    # Sort candidates by highest match percentage first
    return sorted(ranked, key=lambda r: r["match_percentage"], reverse=True)

def connect_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=150,
        num_ctx=2048,
        raising=False,
    )

def explain_candidate(llm, job_description, candidate):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful recruitment assistant."
        " Use ONLY the provided resume passages as evidence."
        " Do not assume missing credentials. "
        "If something is not stated, say 'Not found in provided passages'."
        " Do not infer or evaluate sensitive personal characteristics. "
        "Do not make a final hiring decision."
        " Distinguish what matches the job from what is missing. "
        "Give a brief, clear and factual explanation. "
        "Use these headings: Matching skills; "
        "Missing / unverified requirements; "
        "Relevant experience; Recruiter review notes. "
        "Do not invent a suitability percentage."),
        ("human", "Job description:\n{job_description}\n\nResume file: {candidate}\n\nRetrieved resume passages:\n{context}\n\nAnalyze this resume using only the passages above.")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "job_description": job_description,
        "candidate": candidate["candidate"],
        "context": "\n\n---\n\n".join(candidate["passages"]),
    })