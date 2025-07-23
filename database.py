# app/database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# 데이터베이스 파일 경로 설정
DATABASE_URL = "sqlite:///./feedback.db"

# 데이터베이스 파일이 위치할 디렉토리 생성
os.makedirs(os.path.dirname(DATABASE_URL.split("///")[1]), exist_ok=True)

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # SQLite는 이 옵션이 필요합니다.
)

# 데이터베이스 세션을 생성하기 위한 클래스
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 모델 클래스가 상속받을 기본 클래스
Base = declarative_base()

# --- 데이터베이스 테이블 모델 정의 ---

class Diagnosis(Base):
    """진단 기록을 저장하는 테이블"""
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    crop_name = Column(String)
    predicted_disease = Column(String)
    summary = Column(String, nullable=True) # 질병 요약 정보
    solution = Column(String, nullable=True) # 해결 방안 정보
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Feedback 테이블과의 관계 설정
    feedback = relationship("Feedback", back_populates="diagnosis", uselist=False)


class Feedback(Base):
    """사용자 피드백을 저장하는 테이블"""
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"))
    is_correct = Column(Boolean)
    user_provided_disease = Column(String, nullable=True) # 사용자가 제공한 실제 병명
    feedback_timestamp = Column(DateTime, default=datetime.utcnow)

    # Diagnosis 테이블과의 관계 설정
    diagnosis = relationship("Diagnosis", back_populates="feedback")


# 데이터베이스 테이블을 생성하는 함수
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)