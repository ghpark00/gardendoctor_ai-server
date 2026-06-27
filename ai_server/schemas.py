# app/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional

class DiseaseInfo(BaseModel):
    """
    config.py에 정의된 개별 질병의 상세 정보를 담는 모델
    """
    name: str = Field(..., description="진단된 병명 또는 상태", example="단호박흰가루병")
    summary: Optional[str] = Field(None, description="질병에 대한 요약 정보", example="잎과 열매에 흰 가루를 뿌린 듯한 증상이 나타납니다.")
    solution: Optional[str] = Field(None, description="질병에 대한 해결 방안", example="하우스 내 환기를 잘 시키고, 관련 약제를 살포합니다.")

# --- 여기부터 수정 ---

class DiagnosisResult(BaseModel):
    """
    최종 진단 결과 응답 모델
    """
    diagnosis_id: int = Field(..., description="데이터베이스에 저장된 진단 기록의 고유 ID", example=1)
    filename: str = Field(..., description="사용자가 업로드한 파일 이름", example="pumpkin_test.jpg")
    confidence: float = Field(..., description="AI 모델의 예측 신뢰도 (0 ~ 100%)", example=98.7)
    disease_info: DiseaseInfo = Field(..., description="진단된 질병의 상세 정보")

class FeedbackRequest(BaseModel):
    """
    피드백 제출 시 요청 본문 모델 (개선된 버전)
    """
    is_satisfied: bool = Field(..., description="AI의 진단 결과에 만족했는지 여부 (True: 만족, False: 불만족)", example=True)
    comment: Optional[str] = Field(None, description="추가적인 의견이나 코멘트", example="이미지가 좀 흐렸는데 잘 맞춘 것 같아요.")

class SuccessResponse(BaseModel):
    """
    단순 성공 메시지 응답 모델
    """
    message: str = Field(..., description="처리 성공 메시지", example="피드백이 성공적으로 저장되었습니다.")


class AnalysisRequest(BaseModel):
    image_url: str

    
class ErrorResponse(BaseModel):
    """
    오류 발생 시 응답 모델
    """
    error: str = Field(..., description="오류 메시지")

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# --- 챗봇 세션 관련 모델들 ---

class ChatMessage(BaseModel):
    """챗봇 메시지 모델"""
    id: int = Field(..., description="메시지 고유 ID", example=1)
    role: str = Field(..., description="메시지 역할", example="user")  # "user" 또는 "assistant"
    query: str = Field(..., description="메시지 내용", example="토마토 잎이 노랗게 변하는 이유는 뭔가요?")
    timestamp: str = Field(..., description="메시지 작성 시간", example="2025-01-15T10:30:00")

class ChatSessionResponse(BaseModel):
    """챗봇 세션 응답 모델 (첫 메시지를 제목으로 사용)"""
    id: int = Field(..., description="세션 고유 ID", example=1)
    query: Optional[str] = Field(None, description="첫 번째 사용자 질문 (제목 역할)", example="토마토 잎이 노랗게 변해요")
    created_at: str = Field(..., description="세션 생성 시간", example="2025-01-15T10:30:00")
    updated_at: str = Field(..., description="세션 수정 시간", example="2025-01-15T11:45:00")
    message_count: int = Field(0, description="대화 메시지 수", example=5)
    
    @validator('query')
    def truncate_query(cls, v):
        """제목용 쿼리를 30자로 제한"""
        if v and len(v) > 30:
            return v[:30] + "..."
        return v

class ChatSessionWithMessages(BaseModel):
    """메시지가 포함된 챗봇 세션 모델"""
    id: int = Field(..., description="세션 고유 ID", example=1)
    created_at: str = Field(..., description="세션 생성 시간", example="2025-01-15T10:30:00")
    updated_at: str = Field(..., description="세션 수정 시간", example="2025-01-15T11:45:00")
    messages: list[ChatMessage] = Field(
        [], 
        description="대화 메시지 목록",
        example=[
            {
                "id": 1,
                "role": "user",
                "query": "토마토 잎이 노랗게 변하는 이유는 뭔가요?",
                "timestamp": "2025-01-15T10:30:00"
            },
            {
                "id": 2,
                "role": "assistant", 
                "query": "토마토 잎이 노랗게 변하는 주요 원인은 다음과 같습니다:\n\n1. **질소 부족**: 아래쪽 잎부터 노랗게 변함\n2. **물 부족**: 토양이 건조할 때\n3. **병해**: 시들음병이나 역병\n\n어떤 부위의 잎이 먼저 노랗게 변하시나요?",
                "timestamp": "2025-01-15T10:30:45"
            }
        ]
    )

class ChatRequest(BaseModel):
    """챗봇 대화 요청 모델"""
    session_id: Optional[int] = Field(None, description="기존 세션 ID (없으면 새 세션 생성)", example=1)
    query: str = Field(..., description="사용자 질문", example="토마토 잎이 노랗게 변하는 이유는 뭔가요?")