import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langgraph_agent_react import run_agent

# 환경변수 로드
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flutter 앱 주소를 명시해도 좋음 (보안상)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    answer = await run_agent(req.query)
    return {"answer": answer}
