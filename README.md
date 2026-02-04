📘 Enterprise Knowledge Assistant (RAG)

A Retrieval-Augmented Generation (RAG)–based system designed to answer natural-language questions over enterprise policy documents using semantic search and vector embeddings.
The system retrieves relevant document context and generates grounded, non-hallucinated answers, making it suitable for real-world enterprise knowledge management.

🔍 Problem Statement

Large organizations store critical knowledge across policy documents, guidelines, and manuals. Traditional keyword search often fails to capture semantic meaning, while large language models (LLMs) can hallucinate answers if not grounded in source data.

This project addresses these challenges by:

enabling semantic retrieval over enterprise documents

ensuring answers are based strictly on retrieved content

providing a modular architecture that can later integrate cloud-based LLMs

🧠 Solution Overview

The system implements a Retrieval-Augmented Generation (RAG) pipeline:

Enterprise documents are ingested and preprocessed

Documents are split into overlapping text chunks

Each chunk is converted into a semantic vector embedding

Embeddings are stored in a FAISS vector database

User queries are embedded and matched using similarity search

Retrieved context is used to generate a grounded answer

This design avoids hallucination and ensures traceability to source documents.

🧱 Architecture
Enterprise Documents (.txt)
        ↓
Text Cleaning & Chunking
        ↓
Embedding Generation
        ↓
FAISS Vector Store
        ↓
Semantic Retrieval
        ↓
Grounded Answer Generation

🛠️ Technology Stack

Programming Language: Python

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Vector Database: FAISS

Libraries: LangChain (community modules), Hugging Face, NumPy

Environment: Local execution (no API dependency)

📂 Project Structure
Enterprise-Knowledge-Assistant/
│
├── data/
│   ├── hr_policy.txt
│   ├── leave_policy.txt
│   ├── it_guidelines.txt
│
├── src/
│   ├── rag_pipeline.py     # Document ingestion + embedding + indexing
│   ├── retrieve.py         # Semantic retrieval
│   └── answer.py           # Grounded answer generation
│
├── faiss_index/            # Saved vector database
├── requirements.txt
└── README.md

🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Build the vector index
python src/rag_pipeline.py


This step:

loads documents

chunks text

generates embeddings

stores them in FAISS

3️⃣ Test semantic retrieval
python src/retrieve.py


Example query:

Can I install software on my company laptop?

4️⃣ Generate a grounded answer
python src/answer.py


Example output:

Based on company policy:
Employees are prohibited from installing unauthorized software.
All data must be backed up to the company cloud storage weekly.

Sources:
data/it_guidelines.txt

✅ Key Features

Semantic (meaning-based) document search

No hallucination: answers derived strictly from retrieved context

Source traceability for every response

Modular design for future LLM or cloud integration

Fully functional without paid APIs

📈 Future Enhancements

Integration with cloud LLMs (AWS Bedrock / OpenAI)

Web UI using Streamlit or FastAPI

Support for PDF and DOCX documents

Role-based access control for enterprise users

Query result ranking and confidence scoring

🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

Retrieval-Augmented Generation (RAG) architectures

Vector embeddings and semantic similarity search

FAISS-based vector databases

Designing AI systems that prioritize correctness and trust

Bridging data engineering with applied AI workflows

📄 License

This project is for educational and demonstration purposes.

🙌 Acknowledgements

Inspired by real-world enterprise knowledge management challenges and modern applied AI system design patterns.
