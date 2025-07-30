# 베이스 이미지로 python 3.9 사용 (필요시 버전 조정 가능)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 OS 라이브러리 설치 (OpenCV, Pillow 등 이미지 처리용)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# AI 모델 파일 복사 (app/core/predict.py 에서 참조하는 경로 맞춤)
# 만약 모델 파일이 별도 경로에 있다면, 해당 위치로 변경 필요
# 여기서는 프로젝트 내 app/ai_models 폴더 가정
# (이미 COPY . . 했으므로 별도 명령 없어도 포함됨)
# 만약 외부에 있으면 따로 COPY 명령 추가 필요

# uvicorn 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
