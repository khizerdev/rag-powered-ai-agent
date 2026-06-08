# langgraph_agent.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from vector_store import search_todos, get_vectorstore_count
from database import SessionLocal
from models import Todo
import os
from datetime import datetime

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
)

# --- State Definition ---
class TodoState(TypedDict):
    user_input: str
    intent: str        # search, save, delete, edit, summarize, schedule, suggest
    response: str

# --- Node: Detect Intent ---
def detect_intent(state: TodoState) -> TodoState:
    """Classify user intent."""

    prompt = ChatPromptTemplate.from_template("""
You are classifying user input for a todo app.

Classify this input as exactly one word:
- "search"    → user asks a question about todos
- "save"      → user adds a new todo
- "delete"    → user removes a todo
- "edit"      → user modifies a todo
- "summarize" → user wants overview of all todos
- "schedule"  → user wants smart scheduling advice
- "suggest"   → user wants proactive suggestions

Input: {user_input}

Reply with only one word.
""")

    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"user_input": state["user_input"]}).strip().lower()

    if intent not in ["search", "save", "delete", "edit", "summarize", "schedule", "suggest"]:
        intent = "search"

    print(f"[Agent] Intent: {intent}")
    return {**state, "intent": intent}

# --- Node: Search ---
def handle_search(state: TodoState) -> TodoState:
    """Answer questions using ChromaDB."""
    relevant_todos = search_todos(state["user_input"], n_results=5)

    if not relevant_todos:
        return {**state, "response": "I couldn't find any relevant todos."}

    todos_text = "\n".join(f"- {todo}" for todo in relevant_todos)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant managing a todo list.

Here are the relevant tasks:
{todos}

Answer this question: {question}

Be concise and direct.
""")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "todos": todos_text,
        "question": state["user_input"]
    })

    return {**state, "response": response}

# --- Node: Save ---
def handle_save(state: TodoState) -> TodoState:
    """Save new todos."""
    from ai import extract_todos
    from vector_store import add_todo_to_vectorstore

    todos = extract_todos(state["user_input"])
    db = SessionLocal()
    saved = []

    for item in todos:
        todo = Todo(
            task=item["task"],
            priority=item["priority"],
            due_date=item["due_date"]
        )
        db.add(todo)
        db.commit()
        db.refresh(todo)
        add_todo_to_vectorstore(todo.id, todo.task)
        saved.append(item["task"])

    db.close()
    response = f"Saved {len(saved)} todo(s): {', '.join(saved)}"
    return {**state, "response": response}

# --- Node: Delete ---
def handle_delete(state: TodoState) -> TodoState:
    """Delete a todo by matching its text."""
    from vector_store import delete_todo_from_vectorstore

    db = SessionLocal()
    
    # Find todo matching user input
    query = state["user_input"].lower()
    todo = db.query(Todo).filter(
        Todo.task.ilike(f"%{query}%")
    ).first()

    if not todo:
        db.close()
        return {**state, "response": f"Could not find a todo matching '{query}'"}

    task_text = todo.task
    db.delete(todo)
    db.commit()
    delete_todo_from_vectorstore(todo.id)
    db.close()

    return {**state, "response": f"Deleted: '{task_text}'"}

# --- Node: Edit ---
def handle_edit(state: TodoState) -> TodoState:
    """Edit an existing todo."""
    from vector_store import add_todo_to_vectorstore

    db = SessionLocal()

    # Ask Groq to parse: which todo to edit, what's the new text
    prompt = ChatPromptTemplate.from_template("""
Parse this todo edit request and extract:
1. The original todo text (what to find)
2. The new todo text (what to replace with)

Request: {request}

Reply in this exact format:
OLD: [original task]
NEW: [new task]
""")

    chain = prompt | llm | StrOutputParser()
    parsed = chain.invoke({"request": state["user_input"]})

    lines = parsed.split("\n")
    old_task = lines[0].replace("OLD:", "").strip() if len(lines) > 0 else ""
    new_task = lines[1].replace("NEW:", "").strip() if len(lines) > 1 else ""

    if not old_task or not new_task:
        db.close()
        return {**state, "response": "Could not parse edit request. Try: 'Change [old task] to [new task]'"}

    todo = db.query(Todo).filter(
        Todo.task.ilike(f"%{old_task}%")
    ).first()

    if not todo:
        db.close()
        return {**state, "response": f"Could not find todo matching '{old_task}'"}

    todo.task = new_task
    db.commit()
    db.refresh(todo)
    add_todo_to_vectorstore(todo.id, new_task)
    db.close()

    return {**state, "response": f"Updated: '{old_task}' → '{new_task}'"}

# --- Node: Summarize ---
def handle_summarize(state: TodoState) -> TodoState:
    """Summarize all todos by priority."""
    db = SessionLocal()
    todos = db.query(Todo).all()
    db.close()

    if not todos:
        return {**state, "response": "You have no todos yet."}

    # Group by priority
    high = [t.task for t in todos if t.priority == "high"]
    medium = [t.task for t in todos if t.priority == "medium"]
    low = [t.task for t in todos if t.priority == "low"]

    summary = f"""
📊 Todo Summary:

🔴 HIGH PRIORITY ({len(high)}):
{chr(10).join(f"  • {t}" for t in high) if high else "  None"}

🟡 MEDIUM PRIORITY ({len(medium)}):
{chr(10).join(f"  • {t}" for t in medium) if medium else "  None"}

🟢 LOW PRIORITY ({len(low)}):
{chr(10).join(f"  • {t}" for t in low) if low else "  None"}

Total: {len(todos)} todos
"""

    return {**state, "response": summary}

# --- Node: Schedule ---
def handle_schedule(state: TodoState) -> TodoState:
    """Smart scheduling advice."""
    db = SessionLocal()
    todos = db.query(Todo).all()
    db.close()

    if not todos:
        return {**state, "response": "You have no todos to schedule."}

    todos_text = "\n".join(f"- [{t.priority}] {t.task} (due: {t.due_date or 'no date'})" for t in todos)

    prompt = ChatPromptTemplate.from_template("""
You are a productivity assistant. Given these todos, suggest a smart schedule for today/this week.
Consider priorities and due dates. Be practical and encouraging.

Todos:
{todos}

Provide a brief schedule suggestion (3-4 sentences).
""")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"todos": todos_text})

    return {**state, "response": response}

# --- Node: Suggest ---
def handle_suggest(state: TodoState) -> TodoState:
    """Proactive suggestions based on todos."""
    db = SessionLocal()
    todos = db.query(Todo).all()
    db.close()

    if not todos:
        return {**state, "response": "You have no todos. Add some tasks to get started!"}

    high_priority = [t for t in todos if t.priority == "high"]

    if high_priority:
        response = f"⚠️  You have {len(high_priority)} high-priority task(s):\n"
        response += "\n".join(f"  • {t.task}" for t in high_priority)
        response += "\n\nI recommend tackling these first!"
    else:
        response = "✅ No urgent tasks right now. You can work on medium or low priority items."

    return {**state, "response": response}

# --- Routing ---
def route(state: TodoState) -> str:
    return state["intent"]

# --- Build Graph ---
graph = StateGraph(TodoState)

graph.add_node("detect_intent", detect_intent)
graph.add_node("search", handle_search)
graph.add_node("save", handle_save)
graph.add_node("delete", handle_delete)
graph.add_node("edit", handle_edit)
graph.add_node("summarize", handle_summarize)
graph.add_node("schedule", handle_schedule)
graph.add_node("suggest", handle_suggest)

graph.set_entry_point("detect_intent")

graph.add_conditional_edges(
    "detect_intent",
    route,
    {
        "search": "search",
        "save": "save",
        "delete": "delete",
        "edit": "edit",
        "summarize": "summarize",
        "schedule": "schedule",
        "suggest": "suggest",
    }
)

for node in ["search", "save", "delete", "edit", "summarize", "schedule", "suggest"]:
    graph.add_edge(node, END)

app = graph.compile()

# --- Test ---
if __name__ == "__main__":
    test_inputs = [
        "Summarize my todos",
        "What should I do first?",
        "Delete pay electricity bill",
        "Change buy groceries to buy groceries and cook dinner",
        "Suggest what I should focus on",
        "Schedule my tasks for me",
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        result = app.invoke({
            "user_input": user_input,
            "intent": "",
            "response": ""
        })
        print(f"Agent: {result['response']}\n")