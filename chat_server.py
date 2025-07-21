import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langgraph_agent_react import run_agent

# 환경변수 로드
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    answer = await run_agent(req.query)
    return {"answer": answer}
