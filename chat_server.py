# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Path, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import requests

# --- 데이터베이스 및 모델 관련 임포트 ---
import database
import models 
from models import DiagnosisResult, ErrorResponse, AnalysisRequest, FeedbackRequest, SuccessResponse, QueryRequest, ChatResponse
from langgraph_agent_react import run_agent

# --- 수정된 진단 함수 임포트 ---
from core.predict import classify_crop, predict_disease
from core.validation import validate_image_content

# 환경변수 로드 및 DB 테이블 생성
load_dotenv()
database.create_db_and_tables()

app = FastAPI(
    title="계층적 작물 질병 진단 API",
    description="이미지를 업로드하면 AI가 작물을 자동 분류한 후, 해당 작물에 특화된 모델로 질병을 진단합니다.",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "OK", "message": "계층적 작물 질병 진단 서버가 준비되었습니다."}

@app.post(
    "/diagnose",
    summary="이미지 자동 진단 (작물 분류 ➡️ 질병 진단)",
    description="""
    사용자가 이미지 파일을 업로드하면, 시스템이 2단계로 분석합니다:
    1.  **작물 분류**: AI가 먼저 이미지의 작물(토마토, 고추 등)을 자동으로 식별합니다.
    2.  **질병 진단**: 식별된 작물에 특화된 AI 모델을 사용하여 질병을 진단합니다.
    """,
    response_model=DiagnosisResult
)
async def diagnose_image_hierarchical(
    file: UploadFile = File(..., description="진단할 작물 이미지 파일"),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()
    
    is_valid, message = validate_image_content(image_bytes)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"{message} 분석할 수 있는 다른 사진을 업로드해주세요.")
        
    # --- 1단계: 작물 분류 ---
    crop_name, _, error = classify_crop(image_bytes)
    if error:
        raise HTTPException(status_code=500, detail=error)
    if not crop_name:
        raise HTTPException(status_code=400, detail="이미지에서 작물을 식별할 수 없습니다.")

    # --- 2단계: 질병 진단 ---
    disease_info, confidence, error = predict_disease(crop_name, image_bytes)
    if error:
        raise HTTPException(status_code=500, detail=error)

    # --- 진단 결과 DB 저장 ---
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
    db.refresh(new_diagnosis)

    return DiagnosisResult(
        diagnosis_id=new_diagnosis.id,
        filename=file.filename,
        confidence=round(confidence, 2),
        disease_info=models.DiseaseInfo(**disease_info)
    )

@app.post(
    "/diagnose-by-url",
    summary="이미지 URL 자동 진단 (작물 분류 ➡️ 질병 진단)",
    description="이미지 URL을 전송하면, 시스템이 2단계로 분석하여 결과를 DB에 저장합니다.",
    response_model=DiagnosisResult
)
async def diagnose_by_url_hierarchical(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image_bytes = response.content
        Image.open(BytesIO(image_bytes)).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 URL 처리 중 오류 발생: {e}")

    is_valid, message = validate_image_content(image_bytes)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"{message} 분석할 수 있는 다른 사진을 업로드해주세요.")

    # --- 1단계: 작물 분류 ---
    crop_name, _, error = classify_crop(image_bytes)
    if error:
        raise HTTPException(status_code=500, detail=error)
    if not crop_name:
        raise HTTPException(status_code=400, detail="이미지에서 작물을 식별할 수 없습니다.")
        
    # --- 2단계: 질병 진단 ---
    disease_info, confidence, error = predict_disease(crop_name, image_bytes)
    if error:
        raise HTTPException(status_code=500, detail=error)
        
    # --- 진단 결과 DB 저장 ---
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

    return DiagnosisResult(
        diagnosis_id=new_diagnosis.id,
        filename=filename_from_url,
        confidence=round(confidence, 2),
        disease_info=models.DiseaseInfo(**disease_info)
    )

@app.post(
    "/diagnoses/{diagnosis_id}/feedback",
    summary="진단 결과에 대한 피드백 제출",
    description="진단 ID를 통해 특정 진단 결과에 대한 사용자의 만족/불만족 피드백을 저장합니다.",
    response_model=SuccessResponse
)
async def create_feedback(
    diagnosis_id: int = Path(..., description="피드백을 남길 진단 기록의 ID"),
    feedback_data: FeedbackRequest = Body(...),
    db: Session = Depends(get_db)
):
    diagnosis = db.query(database.Diagnosis).filter(database.Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="해당 ID의 진단 기록을 찾을 수 없습니다.")

    existing_feedback = db.query(database.Feedback).filter(database.Feedback.diagnosis_id == diagnosis_id).first()
    if existing_feedback:
        raise HTTPException(status_code=400, detail="이미 해당 진단에 대한 피드백이 존재합니다.")
    
    new_feedback = database.Feedback(
        diagnosis_id=diagnosis_id,
        is_satisfied=feedback_data.is_satisfied,
        comment=feedback_data.comment
    )
    db.add(new_feedback)
    db.commit()

    return SuccessResponse(message="소중한 피드백 감사합니다! AI 모델 개선에 큰 도움이 됩니다.")

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    answer = await run_agent(req.query)
    return ChatResponse(answer=answer)