import chromadb
from sentence_transformers import SentenceTransformer

# ChromaDB stores and searches embeddings for you
# Think of it as: SQLite = stores tasks, ChromaDB = stores meanings

client = chromadb.Client()  # in-memory for now

# A "collection" is like a table, but for vectors
collection = client.create_collection("todos")

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Your todos (normally these come from SQLite) ---
todos = [
    "Call supplier about new stock",
    "Check bike inventory in warehouse",
    "Review quarterly sales numbers",
    "Order replacement parts from vendor",
    "Schedule team meeting for Monday",
    "Update price list for summer collection",
    "Follow up with distributor on delayed shipment",
]

# Encode all todos into vectors
embeddings = model.encode(todos).tolist()  # ChromaDB wants plain lists

# Add to ChromaDB
# Each item needs: id, embedding, document (original text)
collection.add(
    ids=[f"todo_{i}" for i in range(len(todos))],
    embeddings=embeddings,
    documents=todos,
)

print(f"Stored {collection.count()} todos in ChromaDB\n")

# --- Semantic Search ---
def semantic_search(query, n_results=3):
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    return results["documents"][0]  # list of matching todos

# Test it
queries = [
    "inventory work",
    "urgent supplier contact",
    "financial review",
]

for query in queries:
    print(f"Query: '{query}'")
    matches = semantic_search(query)
    for i, match in enumerate(matches, 1):
        print(f"  {i}. {match}")
    print()