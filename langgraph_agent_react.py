import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_tavily import TavilySearch
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing import TypedDict

# 1. 환경 변수
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
print(f"[LangGraph Agent] 사용 모델: {llm.model_name}")

# 2. 툴 정의
# direct_answer 툴 (gpt 직접 응답)
direct_prompt = PromptTemplate(input_variables=["query"], template="{query}")
direct_chain = direct_prompt | llm  # RunnableSequence

direct_tool = Tool(
    name="direct_answer",
    func=lambda q: direct_chain.invoke({"query": q}),
    description="작물 재배나 상식 질문에 대해 직접 응답합니다. 예: '딸기 수확 시기 알려줘', '작물의 광합성 조건은?'"
)

# knowledge_db 툴 (벡터 DB + Fallback)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
knowledge_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def knowledge_with_fallback(query: str) -> str:
    """벡터 DB 검색 후 결과가 부족하면 GPT로 fallback"""
    try:
        response = knowledge_chain.run(query)
        print(f"[DEBUG] 벡터 DB 응답: {response[:100]}...")  # 디버깅용
        
        fallback_phrases = [
           "찾지 못했습니다",
            "정확한 답변을 찾지 못했습니다", 
            "관련 정보가 없습니다",
            "해당 질문에 대한",
            "죄송합니다",
            "내용이 없네요",  
            "정보에", 
            "가져온 정보에"
        ]
        
        # 불충분한 답변이면 GPT로 직접 답변
        if any(phrase in response for phrase in fallback_phrases) or len(response.strip()) < 50:
            print(f"[fallback] knowledge_db 응답 부족 → direct_answer 재시도: {query}")
            
            # 농업 전문 프롬프트로 GPT 답변 생성
            agricultural_prompt = f"""
농업 전문가로서 다음 질문에 대해 상세하고 실용적으로 답변해주세요.

질문: {query}

답변 가이드라인:
1. 농업 전문 지식을 바탕으로 구체적인 정보 제공
2. 재배 방법, 관리 요령, 주의사항 등 실행 가능한 조언
3. 병해충 관련이면 증상, 원인, 방제 방법 포함
4. 전문용어는 쉽게 풀어서 설명
5. 계절별 관리 팁이나 추가 정보가 있다면 포함

답변:"""
            
            gpt_result = direct_chain.invoke({"query": agricultural_prompt})
            final_answer = gpt_result.content if hasattr(gpt_result, 'content') else str(gpt_result)
            print(f"✅ [FALLBACK 완료] GPT 답변 생성됨")
            return f"{final_answer}\n\n💡 *농업 전문 AI 지식을 바탕으로 답변드렸습니다.*"
        print(f"✅ [벡터 DB] 충분한 응답 반환")
        return response
        
    except Exception as e:
        print(f"[ERROR] knowledge_db 오류 → direct_answer로 fallback: {e}")
        fallback_prompt = f"농업 전문가로서 '{query}'에 대해 도움이 되는 정보를 알려주세요."
        fallback_result = direct_chain.invoke({"query": fallback_prompt})
        return fallback_result.content if hasattr(fallback_result, 'content') else str(fallback_result)

# knowledge_db 툴
knowledge_tool = Tool(
    name="knowledge_db",
    func=knowledge_with_fallback,
    description=(
        "전문 PDF 문서와 농사로 데이터에서 관련 정보를 찾아 상세히 요약합니다. "
        "벡터 DB에 충분한 정보가 없으면 농업 전문 AI 지식으로 자동 보완합니다. "
        "작물 재배, 병해충 방제, 농업 기술 등 모든 농업 관련 질문에 대응합니다. "
        "예: '딸기 잎에 흰 반점 생겼어', '토마토 재배법', '작물 병해충 원인과 대응 방법'"
    )
)
# web_search 툴
tavily = TavilySearch(api_key=TAVILY_API_KEY)
web_tool = Tool(name="web_search", func=tavily.run, description=
                "실시간 정보(날씨, 현재 기온, 뉴스 등)를 검색합니다. "
                "예: '오늘 날씨 어때?', '지금 기온이 몇 도야?', '경기도 지역 강수량 예보 알려줘'")

tools = [direct_tool, knowledge_tool, web_tool]

# 3. LangChain ReAct 기반 Agent 구성
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # ReAct 기반 에이전트
    verbose=True
)

# 4. LangGraph용 상태 모델
class AgentState(TypedDict):
    input: str
    output: str

# 5. LangGraph 노드 정의
def run_agent_step(state: AgentState):
    user_input = state["input"]
    response = agent.run(user_input)
    return {"output": response}

# 6. LangGraph 그래프 구성
builder = StateGraph(AgentState)
builder.set_entry_point("agent_executor")
builder.add_node("agent_executor", run_agent_step)
builder.add_edge("agent_executor", END)

agent_graph = builder.compile()

# 7. FastAPI용 함수
async def run_agent(query: str) -> str:
    result = await agent_graph.ainvoke({"input": query})
    return f"{result['output']}"

