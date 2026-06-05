# langchain_test.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: The model (replaces your manual OpenAI client)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
)

# Step 2: The prompt template (replaces your manual f-string)
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant managing a todo list.

Here are the relevant tasks:
{todos}

Answer this question: {question}

Be concise and direct.
""")

# Step 3: Output parser (replaces response.choices[0].message.content)
parser = StrOutputParser()

# Step 4: The chain — connect all 3 with pipe operator
chain = prompt | llm | parser

# Step 5: Run it
result = chain.invoke({
    "todos": "- Call dentist tomorrow\n- Order food\n- Check inventory",
    "question": "Any health related tasks?"
})

print(result)