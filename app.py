from ai import extract_todos
from database import SessionLocal
from models import Todo
from vector_store import add_todo_to_vectorstore

db = SessionLocal()

user_input = """
Tomorrow call dentist.
Order food.
Check inventory.
"""

todos = extract_todos(user_input)

for item in todos:

    todo = Todo(
        task=item["task"],
        priority=item["priority"],
        due_date=item["due_date"]
    )

    db.add(todo)
    db.commit()        # commit immediately so SQLite assigns an id
    db.refresh(todo)   # pull the assigned id back into the todo object

    add_todo_to_vectorstore(todo.id, todo.task)   # sync to ChromaDB

    print(f"Saved: [{todo.id}] {todo.task} ({todo.priority})")

print("\nAll todos saved to SQLite + ChromaDB")