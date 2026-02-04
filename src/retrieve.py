from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load saved vector store
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def retrieve(query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    query = input("Ask a question: ")
    docs = retrieve(query, k=3)

    print("\n--- Top Retrieved Chunks ---\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] Source: {d.metadata.get('source', 'unknown')}")
        print(d.page_content)
        print("-" * 60)
