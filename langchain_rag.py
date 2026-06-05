# langchain_rag.py
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

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant managing a todo list.

Here are the relevant tasks:
{todos}

Answer this question: {question}

Be concise and direct.
""")

parser = StrOutputParser()

chain = prompt | llm | parser


def ask_about_todos_langchain(question: str) -> str:
    # Step 1: Semantic search (your existing ChromaDB code)
    relevant_todos = search_todos(question, n_results=5)

    if not relevant_todos:
        return "I couldn't find any relevant todos for that question."

    # Step 2: Format todos
    todos_text = "\n".join(f"- {todo}" for todo in relevant_todos)

    # Step 3: One line replaces prompt building + Groq call + response parsing
    return chain.invoke({
        "todos": todos_text,
        "question": question
    })


# Test it
questions = [
    "What should I do tomorrow?",
    "Any health related tasks?",
    "What food tasks do I have?",
]

for question in questions:
    print(f"Q: {question}")
    print(f"A: {ask_about_todos_langchain(question)}")
    print()