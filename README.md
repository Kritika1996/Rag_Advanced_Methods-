# Rag_Advanced_Methods-

Advanced RAG Systems: Fusion Retrieval & Semantic Chunking
Two sophisticated Retrieval-Augmented Generation (RAG) systems designed for high-quality document querying. The Fusion system combines keyword and semantic search for robust performance, while the Semantic system chunks documents by meaning for superior contextual accuracy. Both leverage Pinecone for efficient vector storage and retrieval.
Features
🔄 Fusion Retrieval RAG


Hybrid Search Fusion: Combines dense vector search (semantic) and sparse keyword search (BM25) using a tunable alpha parameter for balanced retrieval.


Intelligent Caching: Stores every query and its generated answer in Pinecone for instant response to repeated questions.


Comprehensive Answer Generation: Synthesizes answers from multiple retrieved document chunks by extracting the most relevant parts.


🧠** Semantic Chunking RAG**


Semantic Document Chunking: Uses embedding models to split documents at natural semantic breakpoints, creating chunks that represent complete ideas and topics.


Performance Metrics: Tracks and reports chunking time, retrieval time, and cosine similarity scores for evaluation.


Query Caching: Stores successful queries and their answers in Pinecone for fast future retrieval.


Tech Stack


Python – Core programming language


Pinecone – Vector database for storage and similarity search


Langchain – Framework for document loading and chunking


Google Gemini Embeddings – Embedding model (models/embedding-001)


Rank-BM25 – Keyword search algorithm for sparse retrieval


Langchain-Experimental – Provides the SemanticChunker component


PyPDF2 – PDF text extraction


python-dotenv – Environment variable management
