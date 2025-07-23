# app/core/predict.py
import torch
import torch.nn.functional as F
import io
from torchvision import models, transforms
from PIL import Image
from .config import DISEASE_CLASSES # config.py에서 상세 정보 가져오기

# --- 1. 작물별 모델 및 설정 정보 ---
# 각 작물에 대한 모델 경로와 학습 시 사용한 클래스 이름을 정의합니다.
# 이 정보는 앱에서 호출하는 crop_name과 일치해야 합니다.
MODEL_CONFIG = {
    "tomato": {
        "model_path": r"ai_models\new_tomato_classifier_ver2.pth", 
        "class_names": ['정상', '토마토잎곰팡이병', '토마토황화잎말림바이러스병', '토마토흰가루병'] 
    },
    "strawberry": {
        "model_path": r"ai_models\no_powdery_leaf_strawberry_classifier.pth", 
        "class_names": ['딸기꽃곰팡이병', '딸기모무늬병', '딸기열매흰가루병', '딸기잎마름병', '딸기잿빛곰팡이병', '딸기탄저병', '정상'] # 참외 모델 학습 시 클래스 순서
    },
    "pepper": {
        "model_path": r"ai_models\pepper_classifier_ver1.pth", 
        "class_names": ['고추탄저병', '고추흰가루병', '정상'] 
    },
    "pumpkin": {
        "model_path": r"ai_models\best_pumpkin_classifier_811.pth",
        "class_names": ['단호박점무늬병', '단호박흰가루병', '정상'] # 학습 순서와 반드시 일치해야 함
    },
    "k_melon": {
        "model_path": r"ai_models\best_k_melon_classifier_811.pth", # 참외 모델 경로 (예시)
        "class_names": ['정상', '참외노균병', '참외흰가루병'] # 참외 모델 학습 시 클래스 순서
    },
    
    # 여기에 새로운 작물 모델 정보를 계속 추가할 수 있습니다.
    # "tomato": {
    #     "model_path": r"app/ai_models/tomato_model.pth",
    #     "class_names": [...]
    # },
}

# --- 2. 모델 캐시 및 전역 변수 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE = {} # 로드된 모델을 저장하여 재사용하기 위한 딕셔너리

# 이미지 전처리 파이프라인 (모든 모델이 동일한 전처리를 사용한다고 가정)
preprocess = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(crop_name: str):
    """
    요청된 작물 이름에 해당하는 모델을 로드하고 캐시에 저장합니다.
    """
    # 1. 지원하는 작물인지 확인
    if crop_name not in MODEL_CONFIG:
        return None, None, "지원되지 않는 작물입니다."
    
    # 2. 모델이 이미 캐시에 있으면 캐시된 모델을 반환
    if crop_name in MODEL_CACHE:
        return MODEL_CACHE[crop_name]["model"], MODEL_CACHE[crop_name]["class_names"], None

    # 3. 캐시에 없으면 모델을 파일에서 로드
    config = MODEL_CONFIG[crop_name]
    model_path = config["model_path"]
    class_names = config["class_names"]
    num_classes = len(class_names)

    try:
        # 모델 아키텍처 정의 (EfficientNet B1 기준)
        model = models.efficientnet_b1(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        # 학습된 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval() # 평가 모드로 설정

        # 로드된 모델을 캐시에 저장
        MODEL_CACHE[crop_name] = {"model": model, "class_names": class_names}
        print(f"✅ '{crop_name}' 모델 로드 및 캐싱 완료: {model_path}")
        return model, class_names, None
    except FileNotFoundError:
        error_msg = f"'{model_path}' 경로에 모델 파일이 없습니다."
        print(f"❌ {error_msg}")
        return None, None, error_msg
    except Exception as e:
        error_msg = f"'{crop_name}' 모델 로드 중 오류 발생: {e}"
        print(f"❌ {error_msg}")
        return None, None, "AI 모델을 불러오는 데 실패했습니다."


def predict_disease(crop_name: str, image_bytes: bytes) -> tuple:
    """
    지정된 작물 모델을 사용하여 질병을 예측하고, 상세 정보와 신뢰도를 반환합니다.
    """
    # 1. 작물에 맞는 모델과 클래스 정보 로드
    model, class_names, error = load_model(crop_name)
    if error:
        return None, 0.0, error

    # 2. 이미지 전처리
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return None, 0.0, f"이미지 파일을 처리할 수 없습니다: {e}"

    # 3. 예측 수행
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    # 4. 예측된 클래스 이름으로 config.py에서 상세 정보 찾기
    crop_disease_info = DISEASE_CLASSES.get(crop_name, {})
    final_info = None
    for key, info in crop_disease_info.items():
        if info.get("name") == predicted_class_name:
            final_info = info
            break
    
    if not final_info:
        return None, 0.0, f"'{predicted_class_name}'에 대한 상세 정보를 config.py에서 찾을 수 없습니다."

    return final_info, confidence_score, None
