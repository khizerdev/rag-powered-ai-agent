from database import SessionLocal
from models import Todo
from ai import ask_about_todos

db = SessionLocal()

todos = db.query(Todo).all()

todo_data = []

for todo in todos:
    todo_data.append({
        "task": todo.task,
        "priority": todo.priority,
        "due_date": todo.due_date
    })

answer = ask_about_todos(
    "Summarize my todos.",
    todo_data
)



print(answer)