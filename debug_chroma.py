# debug_chroma.py  (replace contents)
from vector_store import collection, model

queries = [
    "What should I do tomorrow?",
    "Any health related tasks?",
    "What food tasks do I have?",
]

for query in queries:
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    print(f"Query: '{query}'")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"  dist={dist:.4f}  '{doc}'")
    print()