# langgraph_agent.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from vector_store import search_todos
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
)

# --- Step 1: Define State ---
# This is the data that flows between nodes
class TodoState(TypedDict):
    user_input: str       # what the user typed
    intent: str           # "search" or "save"
    response: str         # final answer

# --- Step 2: Define Nodes ---
# Each node is one function that receives state and returns updated state

def detect_intent(state: TodoState) -> TodoState:
    """Decides: is the user asking a question or adding a todo?"""

    prompt = ChatPromptTemplate.from_template("""
You are classifying user input for a todo app.

Classify this input as exactly one word:
- "search"  → user is asking a question about their todos
- "save"    → user is adding a new todo or task

Input: {user_input}

Reply with only one word: search or save
""")

    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"user_input": state["user_input"]}).strip().lower()

    # Safety fallback
    if intent not in ["search", "save"]:
        intent = "search"

    print(f"[Agent] Intent detected: {intent}")
    return {**state, "intent": intent}


def handle_search(state: TodoState) -> TodoState:
    """Answers questions using ChromaDB + Groq."""

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


def handle_save(state: TodoState) -> TodoState:
    """Saves new todos to SQLite + ChromaDB."""
    from ai import extract_todos
    from database import SessionLocal
    from models import Todo
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


# --- Step 3: Routing function ---
def route(state: TodoState) -> str:
    """Tells LangGraph which node to go to next."""
    return state["intent"]   # returns "search" or "save"


# --- Step 4: Build the Graph ---
graph = StateGraph(TodoState)

# Add nodes
graph.add_node("detect_intent", detect_intent)
graph.add_node("search", handle_search)
graph.add_node("save", handle_save)

# Entry point
graph.set_entry_point("detect_intent")

# After detect_intent, route to search or save
graph.add_conditional_edges(
    "detect_intent",
    route,
    {
        "search": "search",
        "save": "save",
    }
)

# Both search and save end the graph
graph.add_edge("search", END)
graph.add_edge("save", END)

# Compile
app = graph.compile()


# --- Step 5: Test it ---
test_inputs = [
    "What should I do tomorrow?",
    "Any health related tasks?",
    "Buy groceries and pay electricity bill",
]

for user_input in test_inputs:
    print(f"\nUser: {user_input}")
    result = app.invoke({
        "user_input": user_input,
        "intent": "",
        "response": ""
    })
    print(f"Agent: {result['response']}")

print("\n" + "="*50)
print("Testing saved todos are searchable...")
print("="*50 + "\n")

result = app.invoke({
    "user_input": "What do I need to buy or pay for?",
    "intent": "",
    "response": ""
})
print(f"User: What do I need to buy or pay for?")
print(f"Agent: {result['response']}")