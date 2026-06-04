from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": """
            You are a task extraction assistant.

            Return ONLY valid JSON.

            Schema:

            [
              {
                "task": "string",
                "date": "string"
              }
            ]
            """
        },
        {
            "role": "user",
            "content": """
            Tomorrow call the dentist and pay electricity bill.
            """
        }
    ]
)

content = response.choices[0].message.content

todos = json.loads(content)

print(todos)