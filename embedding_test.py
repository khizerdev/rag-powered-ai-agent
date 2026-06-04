from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model (downloads once, cached after)
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude

# --- PART 3: Semantic Search simulation ---
print("\n" + "=" * 50)
print("PART 3: Semantic Search")
print("=" * 50)

# Imagine these are todos from your SQLite database
todos = [
    "Call supplier about new stock",
    "Check bike inventory in warehouse",
    "Review quarterly sales numbers",
    "Order replacement parts from vendor",
    "Schedule team meeting for Monday",
    "Update price list for summer collection",
    "Follow up with distributor on delayed shipment",
]

# User's search query
query = "inventory work"   # <-- no exact match anywhere in todos!

print(f"Todos in database: {len(todos)}")
print(f"Search query: '{query}'\n")

# Encode everything
todo_embeddings = model.encode(todos)
query_embedding = model.encode(query)

# Score all todos against the query
results = []
for i, todo in enumerate(todos):
    score = cosine_similarity(query_embedding, todo_embeddings[i])
    results.append((score, todo))

# Sort by score descending
results.sort(key=lambda x: x[0], reverse=True)

print("Results (ranked by semantic similarity):")
print("-" * 50)
for score, todo in results:
    marker = "  ◀ TOP MATCH" if score == results[0][0] else ""
    print(f"  {score:.4f}  {todo}{marker}")