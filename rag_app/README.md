RAG SYSTEM USING ENDEE VECTOR DATABASE

This project implements a Retrieval Augmented Generation(RAG)  pipeline using Endee as the vector database.  
The system retrieves relevant document chunks using semantic similarity and generates answers using a transformer based language model.

Features:
- Uses Endee for vector storage
- Document ingestion with chunking & embeddings
- Semantic retrieval using cosine similarity
- Text generation using a transformer model
- End-to-end RAG pipeline with a single entry point

Architecture Overview:
Raw Documents -> Text Chunking -> Sentence Embeddings -> Endee Vector Database -> Local Vector Cache -> Semantic Retrieval -> LLM-based Answer Generation

Steps:
Step 1 : Clone Endee and Build Server
- git clone https://github.com/EndeeLabs/endee.git
- cd endee
- mkdir build
- cd build
- cmake .. -DUSE_NEON=ON
- make

Step 2: Run Endee
- ./run.sh binary_file=build/ndd-neon-darwin //
Endee runs at : http://0.0.0.0:8080

Step 3: Set up Python Environment
- cd rag_app
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

Step 4 : Ingest Documents
- python -m app.ingestion.ingest

Step 5 : Run the RAG System
- python -m app.main
