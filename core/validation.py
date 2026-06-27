# app/core/validation.py
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path # ◀️ 수정: pathlib 임포트

# ◀️ 수정 시작
# YOLOv8n 모델 경로를 pathlib을 사용하여 설정
model_path = Path("ai_models") / "validation_yolo_ver3.pt"
# Path 객체를 문자열로 변환하여 모델 로드
model = YOLO(str(model_path)) 
# ◀️ 수정 끝

# ... (이하 동일)
RELEVANT_OBJECT_NAMES = [
    'crop_part'
]

def validate_image_content(image_bytes: bytes) -> tuple[bool, str]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        results = model(image, conf=0.1, verbose=False)
        
        detected_names = {model.names[int(c)] for r in results for c in r.boxes.cls}

        for name in detected_names:
            if name in RELEVANT_OBJECT_NAMES:
                print(f"✅ 유효성 검사 성공: 객체 '{name}' 감지.")
                return True, f"'{name}' 객체가 감지되어 분석을 진행합니다."

        print("❌ 유효성 검사 실패: 이미지에서 관련 객체를 찾을 수 없습니다.")
        return False, "이미지에서 작물 관련 객체(식물, 잎, 과일 등)를 찾을 수 없습니다."

    except Exception as e:
        print(f"🔥 이미지 유효성 검사 중 오류 발생: {e}")
        return False, f"이미지 유효성 검사 중 오류가 발생했습니다: {e}"