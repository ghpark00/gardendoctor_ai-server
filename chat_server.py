# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Path, Depends, Body
from sqlalchemy.orm import Session
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from langgraph_agent_react import run_agent
# --- 데이터베이스 및 모델 관련 임포트 추가 ---
import database
import models 
from core.predict import predict_disease
# Pydantic 모델
from models import DiagnosisResult, ErrorResponse, AnalysisRequest, FeedbackRequest, SuccessResponse, QueryRequest, ChatResponse
# --- ✨ 유효성 검사 함수 임포트 ✨ ---
from core.validation import validate_image_content

# 환경변수 로드
load_dotenv()

# --- 앱 시작 시 데이터베이스 테이블 생성 ---
database.create_db_and_tables()

app = FastAPI(
    title="작물별 질병 진단 API (피드백 기능 추가)",
    description="사용자가 선택한 작물에 맞는 AI 모델을 사용하여 질병을 진단하고, 사용자는 결과에 대한 피드백을 제출할 수 있습니다.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flutter 앱 주소를 명시해도 좋음 (보안상)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 데이터베이스 세션을 얻기 위한 Dependency ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "OK", "message": "작물별 질병 진단 서버가 준비되었습니다."}

@app.post(
    "/diagnose/{crop_name}",
    summary="작물별 이미지 파일 진단",
    description="URL 경로에 작물 이름을 명시하고 이미지 파일을 업로드하면, 해당 작물 전용 AI 모델이 진단하고 결과를 DB에 저장합니다.",
    response_model=DiagnosisResult,
    # ... (responses 부분은 기존과 동일)
)
async def diagnose_crop(
    crop_name: str = Path(..., description="진단할 작물의 영문 이름", example="pumpkin"),
    file: UploadFile = File(..., description="진단할 작물 이미지 파일"),
    db: Session = Depends(get_db) # DB 세션 주입
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()
    
    # --- ✨ 1. 이미지 내용 유효성 검사 ✨ ---
    is_valid, message = validate_image_content(image_bytes)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"{message} 분석할 수 있는 다른 사진을 업로드해주세요."
        )
        
    disease_info, confidence, error = predict_disease(crop_name, image_bytes)

    if error:
        # ... (기존 오류 처리 로직과 동일)
        raise HTTPException(status_code=500, detail=error)

    # --- 진단 결과를 DB에 저장 ---
    new_diagnosis = database.Diagnosis(
        filename=file.filename,
        crop_name=crop_name,
        predicted_disease=disease_info["name"],
        summary=disease_info["summary"],   
        solution=disease_info["solution"],
        confidence=round(confidence, 2)
    )
    db.add(new_diagnosis)
    db.commit()
    db.refresh(new_diagnosis) # DB에 저장된 객체 정보(id 포함)를 다시 로드

    # --- Pydantic 모델로 변환하여 반환 (diagnosis_id 포함) ---
    return DiagnosisResult(
        diagnosis_id=new_diagnosis.id,
        filename=file.filename,
        confidence=round(confidence, 2),
        disease_info=models.DiseaseInfo(**disease_info)
    )

@app.post(
    "/diagnose-by-url/{crop_name}",
    summary="작물별 이미지 URL 진단",
    description="이미지 URL을 전송하면, 해당 작물 전용 AI 모델이 진단하고 결과를 DB에 저장합니다.",
    response_model=DiagnosisResult,
    # ... (responses 부분은 기존과 동일)
)
async def diagnose_by_url(
    crop_name: str, 
    request: AnalysisRequest,
    db: Session = Depends(get_db) # DB 세션 주입
):
    # ... (기존 URL 처리 및 이미지 다운로드 로직과 동일)
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image_bytes = response.content
        Image.open(BytesIO(image_bytes)).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 URL 처리 중 오류 발생: {e}")

    # --- ✨ 1. 이미지 내용 유효성 검사 ✨ ---
    is_valid, message = validate_image_content(image_bytes)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"{message} 분석할 수 있는 다른 사진을 업로드해주세요."
        )

    disease_info, confidence, error = predict_disease(crop_name, image_bytes)

    if error:
        # ... (기존 오류 처리 로직과 동일)
        raise HTTPException(status_code=500, detail=error)
        
    # --- 진단 결과를 DB에 저장 ---
    filename_from_url = request.image_url.split("/")[-1]
    new_diagnosis = database.Diagnosis(
        filename=filename_from_url,
        crop_name=crop_name,
        predicted_disease=disease_info["name"],
        summary=disease_info["summary"],   
        solution=disease_info["solution"],
        confidence=round(confidence, 2)
    )
    db.add(new_diagnosis)
    db.commit()
    db.refresh(new_diagnosis)

    # --- Pydantic 모델로 변환하여 반환 (diagnosis_id 포함) ---
    return DiagnosisResult(
        diagnosis_id=new_diagnosis.id,
        filename=filename_from_url,
        confidence=round(confidence, 2),
        disease_info=models.DiseaseInfo(**disease_info)
    )

# --- ✨ 여기에 새로운 피드백 API 추가 ✨ ---
@app.post(
    "/diagnoses/{diagnosis_id}/feedback",
    summary="진단 결과에 대한 피드백 제출",
    description="진단 ID를 통해 특정 진단 결과에 대한 사용자의 피드백(정확 여부, 실제 병명 등)을 저장합니다.",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "해당 ID의 진단 기록을 찾을 수 없음"},
    }
)

async def create_feedback(
    diagnosis_id: int = Path(..., description="피드백을 남길 진단 기록의 ID", example=1),
    feedback_data: FeedbackRequest = Body(...), 
    db: Session = Depends(get_db)
):
    """
    - **diagnosis_id**: 피드백을 남기고자 하는 진단 기록의 고유 ID입니다.
    - **feedback_data**: 피드백 내용을 담은 JSON 데이터입니다.
    - `is_correct` (bool): AI 진단이 맞았는지 여부
    - `correct_disease_name` (str, optional): 틀렸을 경우, 실제 병명
    """
    # 1. 해당 ID의 진단 기록이 있는지 확인
    diagnosis = db.query(database.Diagnosis).filter(database.Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="해당 ID의 진단 기록을 찾을 수 없습니다.")

    # 2. 이미 피드백이 있는지 확인 (선택사항: 중복 제출 방지)
    existing_feedback = db.query(database.Feedback).filter(database.Feedback.diagnosis_id == diagnosis_id).first()
    if existing_feedback:
        raise HTTPException(status_code=400, detail="이미 해당 진단에 대한 피드백이 존재합니다.")
    
    # 3. 새 피드백 기록 생성 및 저장
    new_feedback = database.Feedback(
        diagnosis_id=diagnosis_id,
        is_correct=feedback_data.is_correct,
        user_provided_disease=feedback_data.correct_disease_name
    )
    db.add(new_feedback)
    db.commit()

    return SuccessResponse(message="피드백이 성공적으로 저장되었습니다! 추후 AI모델 개발에 보탬이 됩니다.")

#  -- 챗봇 --
@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    answer = await run_agent(req.query)
    return {"answer": answer}