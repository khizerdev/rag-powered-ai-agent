# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph_agent import app as agent_app

# FastAPI app
app = FastAPI(title="Todo Agent API")

# Allow React frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class UserMessage(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str

# API endpoint
@app.post("/chat")
async def chat(user_msg: UserMessage) -> AgentResponse:
    """Send a message to the todo agent."""
    result = agent_app.invoke({
        "user_input": user_msg.message,
        "intent": "",
        "response": ""
    })
    return AgentResponse(response=result["response"])

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)