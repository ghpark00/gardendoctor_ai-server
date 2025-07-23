# app/core/validation.py
from ultralytics import YOLO
from PIL import Image
import io

# YOLOv8n 모델 로드 (사전 학습된 범용 모델)
# 실제 서비스에서는 '잎', '줄기' 등 특정 객체를 학습시킨 커스텀 모델 사용을 권장합니다.
model = YOLO('ai_models\yolov8n.pt') 

# COCO 데이터셋에서 식물, 과일, 채소와 관련 있는 객체 목록
# 커스텀 모델을 사용한다면 ['leaf', 'stem', 'fruit'] 와 같이 변경할 수 있습니다.
RELEVANT_OBJECT_NAMES = [
    'potted plant', 'apple', 'orange', 'banana', 'broccoli', 'carrot', 'vase'
]

def validate_image_content(image_bytes: bytes) -> tuple[bool, str]:
    """
    YOLOv8 모델을 사용하여 이미지에 농작물 관련 객체가 있는지 확인합니다.

    Args:
        image_bytes: 검사할 이미지의 바이트 데이터입니다.

    Returns:
        A tuple containing:
        - bool: 관련 객체가 있으면 True, 없으면 False.
        - str: 감지된 객체 이름 또는 실패 메시지.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        results = model(image, verbose=False) # 모델 추론 (상세 로그 비활성화)

        # 탐지된 모든 객체 이름을 추출
        detected_names = {model.names[int(c)] for r in results for c in r.boxes.cls}

        # 관련 객체가 하나라도 있는지 확인
        for name in detected_names:
            if name in RELEVANT_OBJECT_NAMES:
                print(f"✅ 유효성 검사 성공: 객체 '{name}' 감지.")
                return True, f"'{name}' 객체가 감지되어 분석을 진행합니다."

        print("❌ 유효성 검사 실패: 이미지에서 관련 객체를 찾을 수 없습니다.")
        return False, "이미지에서 작물 관련 객체(식물, 잎, 과일 등)를 찾을 수 없습니다."

    except Exception as e:
        print(f"🔥 이미지 유효성 검사 중 오류 발생: {e}")
        # 유효성 검사 중 오류 발생 시, 분석을 중단하고 사용자에게 알림
        return False, f"이미지 유효성 검사 중 오류가 발생했습니다: {e}"