import os
import sys
import re
import uuid
import json
from datetime import datetime
import time
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

def clean_text(text: str) -> str:
    """
    Clean PDF extracted text:
    - Normalize spaces and remove invisible characters
    - Merge single line breaks into spaces, preserve double line breaks as paragraphs
    """
    if not text:
        return ""
    text = text.replace("\xa0", " ") 
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)  
    text = re.sub(r"\n{2,}", "<PARA_BREAK>", text)  
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)  
    text = text.replace("<PARA_BREAK>", "\n\n") 
    return text.strip()

print("Import successful!")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("Error: Missing GEMINI_API_KEY or PINECONE_API_KEY in environment variables!")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fusion-documents"


if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# --- Functions ---
def encode_pdf_and_upsert(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Upsert into Pinecone
    for i, doc in enumerate(chunks):
        doc_text = clean_text(doc.page_content)
        vector = embeddings.embed_query(doc.page_content)
        index.upsert([(str(i), vector, {
            "text": doc_text,
            "type": "document_chunk",  
            "chunk_index": i
            })])
        
    chunks[i].page_content = doc_text
    return chunks

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    tokenized = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized)


def fusion_retrieval(bm25: BM25Okapi, query: str, documents: List[Document], k: int = 5, alpha: float = 0.5) -> Tuple[List[Document], List[float]]:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_vector = embeddings.embed_query(query)
    
    # Pinecone vector search
    vector_results = index.query(vector=query_vector, top_k=len(documents), include_metadata=True)
    vector_scores = np.array([match['score'] for match in vector_results['matches']])
    
    # BM25 keyword search
    bm25_scores = bm25.get_scores(query.split())
    
    # Normalize
    vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-9)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    
    # Fusion
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    top_idx = np.argsort(combined_scores)[::-1][:k]
    
    top_docs = [documents[i] for i in top_idx]
    top_scores = [combined_scores[i] for i in top_idx]
    
    return top_docs, top_scores

def store_query_results(query: str, answer: str, retrieved_docs: List[Document],
                        fusion_scores: List[float], similarity_scores: List[float],
                        alpha: float):
    """Save query + first answer in Pinecone"""

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_vector = embeddings.embed_query(query)

    metadata = {
        "query": query,
        "answer": answer,
        "retrieved_docs": [doc.page_content[:200] for doc in retrieved_docs],
        "fusion_scores": json.dumps(fusion_scores),        
        "similarity_scores": json.dumps(similarity_scores), 
        "alpha": alpha,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "query_answer"
    }

    uid = str(uuid.uuid4())
    index.upsert([(uid, query_vector, metadata)])

def retrieve_previous_answer(query: str, threshold: float = 0.85) -> Dict[str, Any]:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_vector = embeddings.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True,
        filter={"type": "query_answer"}
    )

    if results['matches'] and results['matches'][0]['score'] >= threshold:
        match = results['matches'][0]
        metadata = match['metadata']
        return {
            "answer": metadata.get("answer"),
            "fusion_scores": json.loads(metadata.get("fusion_scores", "[]")),
            "similarity_scores": json.loads(metadata.get("similarity_scores", "[]")),
            "score": match['score']
        }
    return None

def extract_relevant_answer(query: str, document_content: str, max_length: int = 300) -> str:
    """Extract the most relevant part of the document for the query"""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', document_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return document_content[:max_length] + "..." if len(document_content) > max_length else document_content
    
    # Find sentences most relevant to the query
    query_words = set(query.lower().split())
    scored_sentences = []
    
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        common_words = query_words.intersection(sentence_words)
        score = len(common_words)
        
        # Also check for partial matches and important keywords
        if score == 0:
            for q_word in query_words:
                if any(q_word in s_word for s_word in sentence_words):
                    score += 0.5
        
        scored_sentences.append((sentence, score))
    
    # Sort by relevance score
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Build answer from top sentences
    answer_parts = []
    total_length = 0
    
    for sentence, score in scored_sentences:
        if score > 0 and total_length + len(sentence) <= max_length:
            answer_parts.append(sentence)
            total_length += len(sentence)
        if total_length >= max_length:
            break
    
    if not answer_parts:
        # Fallback: take beginning of document
        return document_content[:max_length] + "..." if len(document_content) > max_length else document_content
    
    return " ".join(answer_parts)

def generate_comprehensive_answer(query: str, retrieved_docs: List[Document]) -> str:
    """Generate a comprehensive answer from multiple retrieved documents"""
    if not retrieved_docs:
        return "No relevant information found in the document."
    
    # Extract relevant parts from each document
    relevant_extracts = []
    for doc in retrieved_docs:
        clean_content = clean_text(doc.page_content)
        extract = extract_relevant_answer(query, clean_content)
        relevant_extracts.append(extract)
    
    # Combine and deduplicate
    unique_extracts = []
    seen_content = set()
    
    for extract in relevant_extracts:
        # Simple deduplication
        if extract not in seen_content and len(extract) > 30:
            unique_extracts.append(extract)
            seen_content.add(extract)
    
    if not unique_extracts:
        return "I found some information, but it may not directly answer your question. Please try rephrasing your query."
    
    # Create a coherent answer
    if len(unique_extracts) == 1:
        return unique_extracts[0]
    else:
        return "Based on the document, here's what I found:\n\n" + "\n\n".join(
            f"• {extract}" for extract in unique_extracts[:3]  # Limit to top 3
        )

# --- Class for RAG ---
class FusionRetrievalRAG:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.documents = encode_pdf_and_upsert(pdf_path, chunk_size, chunk_overlap)
        self.bm25 = create_bm25_index(self.documents)

    def query(self, q: str, k: int = 5, alpha: float = 0.5) -> Dict[str, Any]:
        start_time = time.time()
        # First check if we have a cached answer
        cached_result = retrieve_previous_answer(q)
        if cached_result and cached_result["answer"]:
            end_time = time.time()
            return {
                "answer": cached_result["answer"],
                "source_documents": ["Retrieved from previous query"],
                "fusion_scores": [1.0],
                "similarity_scores": [cached_result["score"]],
                "documents_count": 1,
                "cached": True,
                "retrieval_time": end_time - start_time
            }
        
        # Normal processing if no cache
        top_docs, fusion_scores = fusion_retrieval(self.bm25, q, self.documents, k, alpha)
        answer = generate_comprehensive_answer(q, top_docs)  # ← This generates the answer
        
        # Calculate similarity scores
        similarity_scores = []
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_vec = embeddings.embed_query(q)
        
        for doc in top_docs:
            doc_vec = embeddings.embed_query(doc.page_content)
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarity_scores.append(sim)
        
        #
        store_query_results(q, answer, top_docs, fusion_scores, similarity_scores, alpha)
        end_time = time.time()
        return {
            "answer": answer,
            "source_documents": [doc.page_content[:200] + "..." for doc in top_docs],
            "fusion_scores": fusion_scores,
            "similarity_scores": similarity_scores,
            "documents_count": len(top_docs),
            "cached": False,
            "retrieval_time": end_time - start_time
        }
# --- CLI / Script Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fusion Retrieval RAG with Pinecone + Gemini")
    parser.add_argument('--pdf', type=str, default="Copy of Siam Website Content--.pdf",
                    help="Path to PDF file")
    parser.add_argument('--queries', type=str,nargs="*", default=[
        "leadership team of siam computing",
        "what is GTP program",
        "what clients are saying about siam computing and its mvp development over gtp program?",
        "why i need siam computing? how they gain trust from user and what promise?",
        "can you list that what are the developments siam computing offering? does siam involved developing in real estate?",
        "can you list that what are the developments siam computing offering? does siam involved developing chatbots in real estate use cases?",
        "what are the available use cases for chatbots in health care and education domains?",
        "twitter contact of siam computing?",
        "what are core values in siam and what about disjoint production?",
        "In 2020 what happened to siam computing?",
        "instagram contact of siam computing",
        "how to reach out siam and what is the impact of gtp program?",
        "tell me the founder of siam and the associated core values, also how navya care connected with siam?",
        "Worldwide location of siam?"
    ],
                        help="Query to ask")
    parser.add_argument('--top_k', nargs="*", type=int, default=[5,10], help="List of top_k values to retrieve")
    parser.add_argument('--alpha', type=float, default=0.5, help="Fusion alpha 0-1")
    parser.add_argument('--max_words', type=int, default=300, help="Maximum words in generated answer")
    args = parser.parse_args()
    
    retriever = FusionRetrievalRAG(pdf_path=args.pdf)
    
    
    # Loop through queries
    for query in args.queries:
        first_retrieval_done = False  
    
        for k in args.top_k:
            start_time = time.time()
            result = retriever.query(q=query, k=k, alpha=args.alpha)
            end_time = time.time()
            result["retrieval_time"] = end_time - start_time
            
            # Truncate answer if needed
            answer_words = result["answer"].split()
            if len(answer_words) > args.max_words:
                result["answer"] = " ".join(answer_words[:args.max_words]) + "..."
            
            # Store default parser question with its first retrieved answer
            if not first_retrieval_done:
                store_query_results(
                    query=query,
                    answer=result["answer"],
                    retrieved_docs=[Document(page_content=src) for src in result["source_documents"]],
                    fusion_scores=result["fusion_scores"],
                    similarity_scores=result["similarity_scores"],
                    alpha=args.alpha
                )
                first_retrieval_done = True  
            
            # Display
            answer_text = "\n".join([line.strip() for line in result["answer"].splitlines() if line.strip()])
            answer_text = textwrap.fill(answer_text, width=80)

            print(f"\n🔍 Query: {query}")
            print(f"📊 Top K: {k} | Found {result['documents_count']} relevant documents")
            print(f"📝 Alpha (fusion weight): {args.alpha}")
            print("\n" + "="*60)
            print(f"\n💡 ANSWER (limited to {args.max_words} words):\n")
            print(result["answer"])
            print(f"⏱ Retrieval time: {result['retrieval_time']:.2f} seconds")
            print("\n" + "="*60)
            print("\n📚 SOURCE DOCUMENTS (preview):")
            for i, (source, fusion_score, sim_score) in enumerate(
                zip(result["source_documents"], result["fusion_scores"], result["similarity_scores"]), 1
            ):
                source_text = "\n".join([line.strip() for line in source.splitlines() if line.strip()])
                source_text = textwrap.fill(source_text, width=80)
                print(f"\nSource {i} (Fusion: {fusion_score:.3f}, Retrieval time: {result['retrieval_time']:.2f} seconds, Similarity: {sim_score:.3f}):")
                print(source_text)