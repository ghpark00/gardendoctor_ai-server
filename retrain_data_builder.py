# retrain_data_builder.py

import shutil
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from database import Diagnosis, Feedback, engine # DB 모델

# 재학습용 데이터가 저장될 경로
RETRAIN_ROOT = Path('./retraining_pool')
AUTO_CONFIRMED_DIR = RETRAIN_ROOT / 'auto_confirmed'
NEEDS_REVIEW_DIR = RETRAIN_ROOT / 'needs_review'

# 신뢰도 임계값
HIGH_CONFIDENCE_THRESHOLD = 95.0
LOW_CONFIDENCE_THRESHOLD = 60.0

def build_retraining_dataset():
    """피드백을 기반으로 재학습 데이터를 분류합니다."""
    
    # 폴더 초기화
    if RETRAIN_ROOT.exists():
        shutil.rmtree(RETRAIN_ROOT)
    AUTO_CONFIRMED_DIR.mkdir(parents=True)
    NEEDS_REVIEW_DIR.mkdir(parents=True)

    Session = sessionmaker(bind=engine)
    db = Session()
    
    # 아직 처리되지 않은 모든 피드백을 가져옵니다. (feedback 테이블에 'processed' 컬럼 추가 가정)
    feedbacks = db.query(Feedback).join(Diagnosis).filter(Feedback.processed == False).all()

    print(f"처리할 신규 피드백 {len(feedbacks)}건을 발견했습니다.")
    
    for fb in feedbacks:
        diagnosis = fb.diagnosis
        # 원본 이미지 파일의 경로 (실제 환경에 맞게 수정 필요)
        source_image_path = Path(f"./uploaded_images/{diagnosis.filename}")
        
        if not source_image_path.exists():
            continue

        # 1. 사용자가 만족한 경우
        if fb.is_satisfied:
            # 신뢰도와 관계없이 '자동 확신' 데이터로 분류
            target_dir = AUTO_CONFIRMED_DIR / diagnosis.predicted_disease
            target_dir.mkdir(exist_ok=True)
            shutil.copy(source_image_path, target_dir / source_image_path.name)
            print(f"[자동 확신] {diagnosis.filename} -> {diagnosis.predicted_disease}")
        
        # 2. 사용자가 불만족한 경우
        else:
            # AI 신뢰도가 높았는데 불만족 -> 1순위 검토 대상
            if diagnosis.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                priority = "priority_1"
            # AI 신뢰도가 낮았는데 불만족 -> 2순위 검토 대상
            else:
                priority = "priority_2"
            
            target_dir = NEEDS_REVIEW_DIR / priority
            target_dir.mkdir(exist_ok=True)
            shutil.copy(source_image_path, target_dir / source_image_path.name)
            print(f"[검토 필요 - {priority}] {diagnosis.filename}")
            
        # 처리 완료 표시
        fb.processed = True
    
    db.commit()
    db.close()
    print("\n재학습 데이터셋 구축 완료!")


if __name__ == '__main__':
    # 이 스크립트를 실행하면 피드백을 기반으로 데이터가 분류됩니다.
    build_retraining_dataset()