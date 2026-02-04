from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def answer_question(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)

    # Combine retrieved content
    context = " ".join([doc.page_content for doc in docs])

    # Simple grounded answer logic
    answer = f"Based on company policy:\n{context}"

    return answer, docs


if __name__ == "__main__":
    query = input("Ask a question: ")
    answer, sources = answer_question(query)

    print("\n--- Final Answer ---\n")
    print(answer)

    print("\n--- Sources ---")
    for s in sources:
        print(s.metadata.get("source", "unknown"))
