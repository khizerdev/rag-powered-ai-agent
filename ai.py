# ai.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import search_todos

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"


def extract_todos(text: str) -> list:
    """Convert natural language into structured todo list."""

    prompt = f"""Extract todos from this text and return a JSON array.
Each todo must have: task, priority (high/medium/low), due_date (or empty string).
Write the task as a short but clear action phrase (5-8 words when possible).
Return ONLY the JSON array, no explanation.

Text: {text}

Example output:
[
  {{"task": "Call supplier about new stock", "priority": "high", "due_date": "tomorrow"}},
  {{"task": "Order food for the week", "priority": "medium", "due_date": ""}}
]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown code blocks if model wraps response in them
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    return json.loads(content)


def ask_about_todos(question: str) -> str:
    """Answer a question using semantic search over stored todos."""

    # Step 1: Find relevant todos via ChromaDB semantic search
    relevant_todos = search_todos(question, n_results=5)

    if not relevant_todos:
        return "I couldn't find any relevant todos for that question."

    # Step 2: Format for prompt
    todos_text = "\n".join(f"- {todo}" for todo in relevant_todos)

    # Step 3: Groq answers using only relevant context
    prompt = f"""You are a helpful assistant managing a todo list.

Here are the relevant tasks:
{todos_text}

Answer this question: {question}

Be concise and direct."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content