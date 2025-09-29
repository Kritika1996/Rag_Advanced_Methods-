import time
import os
import sys
import re
import textwrap
import json
import argparse
import uuid
from typing import List, Tuple, Dict, Any
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
print("All imports successful!")

# Load environment variables from a .env file
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("Error: GEMINI_API_KEY or PINECONE_API_KEY not found in environment variables!")
    sys.exit(1)

def clean_text(text: str) -> str:
    """
    Clean PDF extracted text:
    - Normalize Unicode spaces
    - Remove non-printable characters
    - Collapse multiple spaces
    - Replace single newlines with spaces (keep double newlines as paragraph breaks)
    """
    if not text:
        return ""

    # Normalize Unicode spaces (non-breaking space, etc.)
    text = text.replace("\xa0", " ")

    # Remove non-printable characters (form feeds, etc.)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Replace multiple newlines with a placeholder (to preserve paragraphs)
    text = re.sub(r"\n{2,}", "<PARA_BREAK>", text)

    # Replace remaining single newlines with spaces
    text = text.replace("\n", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Restore paragraph breaks
    text = text.replace("<PARA_BREAK>", "\n\n")

    return text.strip()

def read_pdf_to_string(pdf_path="Copy of Siam Website Content--.pdf"):
    """
    Read a PDF file and return its text content as a single string.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        sys.exit(1)


class SemanticChunkingRAG:
    """
    A class to handle the Semantic Chunking RAG process for document chunking and query retrieval.
    This version uses Pinecone as the vector database and includes evaluation metrics.
    """

    def __init__(self, path, n_retrieved=10, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Initializes the SemanticChunkingRAG by encoding the content using a semantic chunker.

        Args:
            path (str): Path to the PDF file to encode.
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
            embeddings: Embedding model to use.
            breakpoint_type (str): Type of semantic breakpoint threshold.
            breakpoint_amount (float): Amount for the semantic breakpoint threshold.
        """
        print("\n--- Initializing Semantic Chunking RAG with Pinecone ---")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = "semantic-chunking-rag"
        

        # Create Pinecone index if it doesn't exist
        if self.index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
       
        self.index = self.pc.Index(self.index_name)
        

        # Read PDF content directly
        content = read_pdf_to_string(path)
        print(f"PDF loaded successfully. Content length: {len(content)} characters")

        # Use provided embeddings or initialize Gemini embeddings
        if embeddings:
            self.embeddings = embeddings
        else:
            # Initialize Gemini embeddings with the correct parameter name
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY  
            )

        # Initialize the semantic chunker
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )

        # Measure time for semantic chunking
        start_time = time.time()
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Semantic Chunking Time: {self.time_records['Chunking']:.2f} seconds")
        print(f"Created {len(self.semantic_docs)} semantic chunks")

        # Upsert documents to Pinecone
        self._upsert_to_pinecone()
        
        self.n_retrieved = n_retrieved


    def evaluate_retrieval(self, query: str, retrieved_docs: list) -> list:
        """Evaluate retrieval quality using cosine similarity"""
        query_vec = self.embeddings.embed_query(query)
        
        similarities = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            
            doc_vec = self.embeddings.embed_query(content)
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append(sim)
        
        return similarities

    def _upsert_to_pinecone(self):
        """Upsert semantic chunks to Pinecone vector database"""
        print("Upserting documents to Pinecone...")
        start_time = time.time()
        
        vectors_to_upsert = []
        for i, doc in enumerate(self.semantic_docs):
            vector = self.embeddings.embed_query(doc.page_content)
            vectors_to_upsert.append({
                'id': f"chunk_{i}",
                'values': vector,
                'metadata': {
                    'text': doc.page_content,
                    'chunk_index': i,
                    'content_length': len(doc.page_content),
                    'type': 'document_chunk' 
                }
            })
        
        # Upsert in batches of 100
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i+100]
            self.index.upsert(vectors=batch)
        
        upsert_time = time.time() - start_time
        print(f"Pinecone upsert completed in {upsert_time:.2f} seconds")
        print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
    
    def _store_query_in_pinecone(self, query: str, retrieved_docs: list, similarity_scores: list, answer: str = None):
        """Store query and retrieval results in Pinecone"""
        try:
            query_vector = self.embeddings.embed_query(query)

            query_metadata = {
                'query_text': query,
                'cached_answer': answer,
                'text': answer,  
                'timestamp': datetime.utcnow().isoformat(),
                'retrieved_count': len(retrieved_docs),
                'retrieved_chunks': [str(doc.metadata.get('chunk_index', 'unknown')) for doc in retrieved_docs],
                'similarity_scores': json.dumps([float(score) for score in similarity_scores]),
                'type': 'query_answer'}

            self.index.upsert([{
                'id': f"query_{uuid.uuid4().hex[:8]}",
                'values': query_vector,
                'metadata': query_metadata
            }])

            print(f"✅ Query and answer stored in Pinecone: '{query}'")

        except Exception as e:
            print(f"⚠️ Failed to store query in Pinecone: {e}")


    def _retrieve_previous_answer(self, query: str, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Check if this query was already answered and stored"""
        try:
            query_vector = self.embeddings.embed_query(query)

            results = self.index.query(
                vector=query_vector,
                top_k=self.n_retrieved,
                include_metadata=True,
                filter={"type": "query_answer"} 
            )

            for match in results['matches']:
                if match['score'] >= similarity_threshold:
                    metadata = match['metadata']
                    cached_answer = metadata.get('cached_answer', '')
                    if cached_answer:
                        return {
                            "answer": cached_answer,
                            "score": match['score'],
                            "timestamp": metadata['timestamp'],
                            "similarity_scores": json.loads(metadata.get("similarity_scores", "[]"))
                        }
            return None

        except Exception as e:
            print(f"Error retrieving previous answer: {e}")
            return None

        
    def run(self, query, n_retrieved=None, ignore_cache=False):
        # Only use cached answer if not ignoring cache
        cached_result = None
        if not ignore_cache:
            cached_result = self._retrieve_previous_answer(query)
        
        if cached_result and cached_result["answer"]:
            print(f"🎯 Using cached result for: '{query}'")
            dummy_doc = Document(page_content=cached_result["answer"])
            return (
                {'Retrieval': 0.1, 'Chunking': 0.0},
                [dummy_doc],
                [cached_result["score"]]
            )
        
        if n_retrieved:
            self.n_retrieved = n_retrieved
        
        # Measure time for semantic retrieval
        start_time = time.time()
        
        # Get query embedding
        query_vector = self.embeddings.embed_query(query)
        print(f"🔎 Retrieval using top_k={n_retrieved or self.n_retrieved}")
        # Query Pinecone
        results = self.index.query(
        vector=query_vector,
        top_k=self.n_retrieved,
        include_metadata=True,
        include_values=False,
        filter={"type": "document_chunk"}   
    )

        
        # Convert Pinecone results to document format
        semantic_context = []
        for match in results['matches']:
            doc = Document(
            page_content=match['metadata'].get('text', ''),  # ✅ safe fallback
            metadata={
                'score': match['score'],
                'chunk_index': match['metadata'].get('chunk_index', -1),
                'id': match['id'],
                'type': match['metadata'].get('type', 'unknown')
            }
        )
            semantic_context.append(doc)
        
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Pinecone Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Evaluate retrieval quality
        similarity_scores = self.evaluate_retrieval(query, semantic_context)
        
        # Generate a simple answer from the context
        answer = self._generate_answer_from_context(query, semantic_context)
        
        # Store query and results in Pinecone WITH THE ANSWER
        self._store_query_in_pinecone(query, semantic_context, similarity_scores, answer)
        
        # Display the retrieved context with evaluation metrics
        print("\n--- Retrieved Context with Evaluation ---")
        if semantic_context:
            for i, (doc, sim_score) in enumerate(zip(semantic_context, similarity_scores)):
                print(f"Document {i+1}:")
                print(f"Pinecone Score: {doc.metadata['score']:.4f}")
                print(f"Similarity Score: {sim_score:.4f}")
                print(f"Content: {doc.page_content[:200]}...\n" + "-"*50)
            
            # Print summary metrics
            avg_pinecone_score = np.mean([doc.metadata['score'] for doc in semantic_context])
            avg_similarity = np.mean(similarity_scores)
            print(f"\n📊 Summary Metrics:")
            print(f"Average Pinecone Score: {avg_pinecone_score:.4f}")
            print(f"Average Similarity Score: {avg_similarity:.4f}")
            print(f"Total Retrieval Time: {self.time_records['Retrieval']:.2f}s")
            
        else:
            print("No relevant documents found.")
            
        return self.time_records, semantic_context, similarity_scores
    
    def _generate_answer_from_context(self, query: str, retrieved_docs: list) -> str:
        """Generate a simple answer from retrieved context"""
        if not retrieved_docs:
            return "No relevant information found."
        
        # Combine the first few documents to form an answer
        combined_content = " ".join([doc.page_content for doc in retrieved_docs[:2]])
        
        # Simple extraction - take first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', combined_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return combined_content[:300] + "..." if len(combined_content) > 300 else combined_content
        
        # Take first 3-4 sentences as answer
        answer = " ".join(sentences[:4])

        if len(answer) > 300:
            answer = answer[:300] + "..."

        return answer
    
# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG using Pinecone.")
    parser.add_argument("--path", type=str, default="Copy of Siam Website Content--.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--n_retrieved", type=int, default=10,
                        help="Number of chunks to retrieve for each query (default: 10).")
    parser.add_argument("--breakpoint_threshold_type", type=str,
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        default="percentile",
                        help="Type of breakpoint threshold to use for chunking (default: percentile).")
    parser.add_argument("--breakpoint_threshold_amount", type=float, default=90,
                        help="Amount of the breakpoint threshold to use (default: 90).")
    parser.add_argument("--queries", type=str,nargs="+", default=[
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
        "Worldwide location of siam?"],
                        help="Queries to test the retriever.")

    return parser.parse_args()

# Main function to process PDF, chunk text, and test retriever
def main(args):
    results = {}
    # Initialize SemanticChunkingRAG with Pinecone
    semantic_rag = SemanticChunkingRAG(
        path=args.path,
        n_retrieved=args.n_retrieved,
        breakpoint_type=args.breakpoint_threshold_type,
        breakpoint_amount=args.breakpoint_threshold_amount
    )
    
    print("\n📌 Saving default parser queries in Pinecone with first retrieval...")
    
    for query in args.queries:
        print(f"\n\n============================")
        print(f"🔎 Running query: {query}")
        print(f"============================")

        # Top 5
        time_records_5, context_docs_5, similarity_scores_5 = semantic_rag.run(query, n_retrieved=5, ignore_cache=True)
        results[query] = {}
        results[query]["top5"] = (time_records_5, context_docs_5, similarity_scores_5)
        print(f"✅ Saved default query: '{query}' (Top 5)")

        print(f"\n💡 Top 5 Retrieved Documents for query: {query}")
        for i, (doc, score) in enumerate(zip(context_docs_5, similarity_scores_5), 1):
            doc_text = clean_text(doc.page_content)
            doc_text = textwrap.fill(doc_text, width=80)
            print(f"\nDocument {i} (Similarity: {score:.3f}):\n{doc_text}\n{'-'*60}")

        # Top 10
        time_records_10, context_docs_10, similarity_scores_10 = semantic_rag.run(query, n_retrieved=10, ignore_cache=True)
        results[query]["top10"] = (time_records_10, context_docs_10, similarity_scores_10)

        print(f"\n💡 Top 10 Retrieved Documents for query: {query}")
        for i, (doc, score) in enumerate(zip(context_docs_10, similarity_scores_10), 1):
            doc_text = clean_text(doc.page_content)
            doc_text = textwrap.fill(doc_text, width=80)
            print(f"\nDocument {i} (Similarity: {score:.3f}):\n{doc_text}\n{'-'*60}")

    return results
        
if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())