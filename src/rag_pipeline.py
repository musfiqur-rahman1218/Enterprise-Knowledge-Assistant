import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load documents
data_dir = "data"
documents = []

for file_name in os.listdir(data_dir):
    if file_name.endswith(".txt"):
        loader = TextLoader(os.path.join(data_dir, file_name))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Local embeddings (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save vectorstore to disk
vectorstore.save_local("faiss_index")
print("Vector store saved to ./faiss_index")
