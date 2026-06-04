# vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer

# Load once at module level (expensive to reload)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent client — data survives between runs
# Creates a chroma_db/ folder in your project
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("todos")


def add_todo_to_vectorstore(todo_id: int, task: str):
    """Call this whenever a new todo is saved to SQLite."""
    embedding = model.encode(task).tolist()
    
    collection.upsert(           # upsert = insert or update
        ids=[str(todo_id)],
        embeddings=[embedding],
        documents=[task],
        metadatas=[{"todo_id": todo_id}],
    )
    print(f"[VectorStore] Stored embedding for todo #{todo_id}: '{task}'")


def search_todos(query: str, n_results: int = 3, min_score: float = 0.3):
    """
    Semantic search. Returns list of matching task strings.
    min_score filters out weak matches (remember the financial review lesson).
    """
    count = collection.count()
    if count == 0:
        return []
    
    # Don't ask for more results than we have
    n = min(n_results, count)
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
    )
    
    documents = results["documents"][0]
    distances = results["distances"][0]   # lower distance = more similar
    
    # ChromaDB returns L2 distances, not cosine scores.
    # Filter: distance < 1.5 is a reasonable relevance threshold
    filtered = [
        doc for doc, dist in zip(documents, distances)
        if dist < 1.5
    ]
    
    return filtered


def delete_todo_from_vectorstore(todo_id: int):
    """Call this when a todo is deleted from SQLite."""
    collection.delete(ids=[str(todo_id)])
    print(f"[VectorStore] Removed embedding for todo #{todo_id}")


def get_vectorstore_count():
    return collection.count()