# app/core/predict.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from torchvision import models, transforms
from PIL import Image
from .config import DISEASE_CLASSES # config.py에서 상세 정보 가져오기

# --- 1. 작물별 질병 모델 설정 (2단계: Specialist Models) ---
MODEL_CONFIG = {
    "tomato": {
        "model_path": r"ai_models\new_tomato_classifier_ver2.pth", 
        "class_names": ['정상', '토마토잎곰팡이병', '토마토황화잎말림바이러스병', '토마토흰가루병'] 
    },
    "strawberry": {
        "model_path": r"ai_models\no_powdery_leaf_strawberry_classifier.pth", 
        "class_names": ['딸기꽃곰팡이병', '딸기모무늬병', '딸기열매흰가루병', '딸기잎마름병', '딸기잿빛곰팡이병', '딸기탄저병', '정상']
    },
    "pepper": {
        "model_path": r"ai_models\pepper_classifier_ver1.pth", 
        "class_names": ['고추탄저병', '고추흰가루병', '정상'] 
    },
}

# --- 2. 작물 분류 모델 설정 (1단계: Router Model) ---
ROUTER_MODEL_CONFIG = {
    "model_path": r"ai_models\new_finetuned_plant_model.pth",
    "class_names": ['고추', '딸기', '토마토']
}

# --- 3. 작물 이름 매핑 (한글 -> 영문 키) ---
CROP_NAME_MAP_KR_TO_EN = {
    '고추': 'pepper',
    '딸기': 'strawberry',
    '토마토': 'tomato',
}

# --- 4. 모델 캐시 및 전역 변수 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE = {} # 로드된 모델(라우터, 스페셜리스트)을 저장하여 재사용

# 2단계 Specialist 모델을 위한 이미지 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✨ 1단계 Router 모델을 위한 TTA 전용 이미지 전처리 파이프라인 ✨
tta_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.FiveCrop(224) # 중앙, 좌상, 우상, 좌하, 우하 5개 영역으로 자름
])
to_tensor_transform = transforms.ToTensor()
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_router_model():
    """
    1단계 작물 분류 모델(Router Model)을 로드하고 캐시에 저장합니다.
    """
    model_key = "router"
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]["model"], None

    config = ROUTER_MODEL_CONFIG
    model_path = config["model_path"]
    num_classes = len(config["class_names"])

    try:
        model = models.efficientnet_v2_m(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        MODEL_CACHE[model_key] = {"model": model}
        print(f"✅ 'Router' 모델 로드 및 캐싱 완료: {model_path}")
        return model, None
    except FileNotFoundError:
        return None, f"'{model_path}' 경로에 라우터 모델 파일이 없습니다."
    except Exception as e:
        return None, f"라우터 모델 로드 중 오류 발생: {e}"

def load_specialist_model(crop_name: str):
    """
    2단계 작물별 질병 모델(Specialist Model)을 로드하고 캐시에 저장합니다.
    """
    if crop_name not in MODEL_CONFIG:
        return None, None, "지원되지 않는 작물입니다."
    
    if crop_name in MODEL_CACHE:
        return MODEL_CACHE[crop_name]["model"], MODEL_CACHE[crop_name]["class_names"], None

    config = MODEL_CONFIG[crop_name]
    model_path = config["model_path"]
    class_names = config["class_names"]
    num_classes = len(class_names)

    try:
        model = models.efficientnet_b1(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()

        MODEL_CACHE[crop_name] = {"model": model, "class_names": class_names}
        print(f"✅ '{crop_name}' 전문가 모델 로드 및 캐싱 완료: {model_path}")
        return model, class_names, None
    except FileNotFoundError:
        return None, None, f"'{model_path}' 경로에 전문가 모델 파일이 없습니다."
    except Exception as e:
        return None, None, f"'{crop_name}' 전문가 모델 로드 중 오류 발생: {e}"

def classify_crop(image_bytes: bytes) -> tuple:
    """
    1단계: TTA를 사용하여 이미지를 받아 어떤 작물인지 분류하고 영문 키를 반환합니다.
    """
    model, error = load_router_model()
    if error:
        return None, 0.0, error

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # TTA: 10가지 버전의 이미지 생성 (5 crops * 2 flips)
        five_cropped_images = tta_transform(image)
        augmented_images = []
        for crop in five_cropped_images:
            augmented_images.append(crop)
            augmented_images.append(transforms.functional.hflip(crop))

        # TTA: 10개 버전을 하나의 배치로 만들어 정규화
        tta_batch = torch.stack(
            [normalize_transform(to_tensor_transform(img)) for img in augmented_images]
        ).to(DEVICE)

    except Exception as e:
        return None, 0.0, f"이미지 파일을 처리할 수 없습니다: {e}"

    with torch.no_grad():
        # TTA: 10개 버전에 대해 한 번에 예측 수행
        outputs = model(tta_batch)
        probabilities = F.softmax(outputs, dim=1)
        
        # TTA: 10개 버전의 예측 확률을 평균
        mean_probabilities = torch.mean(probabilities, dim=0)
        
        # TTA: 평균 확률에서 최종 예측 결과 도출
        confidence, predicted_idx = torch.max(mean_probabilities, 0)

    predicted_class_name_kr = ROUTER_MODEL_CONFIG["class_names"][predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    crop_name_en = CROP_NAME_MAP_KR_TO_EN.get(predicted_class_name_kr)
    if not crop_name_en:
        return None, 0.0, f"분류된 작물 '{predicted_class_name_kr}'을 처리할 수 없습니다."

    print(f"🔍 1단계 작물 분류 (TTA 적용): '{crop_name_en}' (신뢰도: {confidence_score:.2f}%)")
    return crop_name_en, confidence_score, None

def predict_disease(crop_name: str, image_bytes: bytes) -> tuple:
    """
    2단계: 지정된 작물 모델을 사용하여 질병을 예측합니다.
    """
    model, class_names, error = load_specialist_model(crop_name)
    if error:
        return None, 0.0, error

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # 2단계에서는 TTA를 사용하지 않고 기존의 단일 CenterCrop으로 예측
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return None, 0.0, f"이미지 파일을 처리할 수 없습니다: {e}"

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    crop_disease_info = DISEASE_CLASSES.get(crop_name, {})
    final_info = None

    if crop_name == "others":
        for key, info in crop_disease_info.items():
            if info.get("eng_name") == predicted_class_name:
                final_info = info
                break
    else:
        for key, info in crop_disease_info.items():
            if info.get("name") == predicted_class_name:
                final_info = info
                break
    
    if not final_info:
        return None, 0.0, f"'{predicted_class_name}'에 대한 상세 정보를 찾을 수 없습니다."

    print(f"🎯 2단계 질병 진단: '{final_info.get('name')}' (신뢰도: {confidence_score:.2f}%)")
    return final_info, confidence_score, None