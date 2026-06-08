# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph_agent import app as agent_app
from datetime import datetime
import json
import os

app = FastAPI(title="Todo Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

class UserMessage(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str

class ChatHistory(BaseModel):
    messages: list

def load_chat_history():
    """Load chat history from file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {"messages": []}

def save_chat_history(history):
    """Save chat history to file."""
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@app.post("/chat")
async def chat(user_msg: UserMessage) -> AgentResponse:
    """Send a message to the todo agent."""
    result = agent_app.invoke({
        "user_input": user_msg.message,
        "intent": "",
        "response": ""
    })
    
    # Save to history
    history = load_chat_history()
    history["messages"].append({
        "timestamp": datetime.now().isoformat(),
        "role": "user",
        "text": user_msg.message
    })
    history["messages"].append({
        "timestamp": datetime.now().isoformat(),
        "role": "agent",
        "text": result["response"]
    })
    save_chat_history(history)
    
    return AgentResponse(response=result["response"])

@app.get("/history")
async def get_history() -> ChatHistory:
    """Get all chat history."""
    history = load_chat_history()
    return ChatHistory(messages=history["messages"])

@app.delete("/history")
async def clear_history():
    """Clear chat history."""
    save_chat_history({"messages": []})
    return {"status": "cleared"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)