# test_pipeline.py
from vector_store import add_todo_to_vectorstore, search_todos, get_vectorstore_count

# Simulate what app.py does when saving todos
test_todos = [
    (1, "Call supplier about new stock"),
    (2, "Check bike inventory in warehouse"),
    (3, "Review quarterly sales numbers"),
    (4, "Order replacement parts from vendor"),
    (5, "Schedule team meeting for Monday"),
    (6, "Update price list for summer collection"),
    (7, "Follow up with distributor on delayed shipment"),
]

print("Storing todos in ChromaDB...")
for todo_id, task in test_todos:
    add_todo_to_vectorstore(todo_id, task)

print(f"\nTotal in vector store: {get_vectorstore_count()}\n")

# Test semantic search
queries = ["inventory work", "urgent supplier contact", "financial review"]

for query in queries:
    print(f"Query: '{query}'")
    results = search_todos(query, n_results=3)
    for r in results:
        print(f"  → {r}")
    print()