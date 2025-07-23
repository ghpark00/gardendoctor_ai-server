# app/models.py
from pydantic import BaseModel, Field
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
    피드백 제출 시 요청 본문 모델
    """
    is_correct: bool = Field(..., description="AI의 예측이 정확했는지 여부", example=False)
    correct_disease_name: Optional[str] = Field(None, description="예측이 틀렸을 경우, 실제 병명", example="단호박점무늬병")

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