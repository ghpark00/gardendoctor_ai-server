# gardendoctor-FastAPI
AI Server

# 환경변수 키 추가 # .env 파일 만들어서 복사&붙여넣기
# OpenAI API 키
# 노션 참고

# 해당 디렉토리(프로젝트 디렉토리)로 이동한 후  터미널에 
python -m venv ai-env

# 패키지 설치
pip install -r requirements.txt

# 벡터 DB 생성 (최초 1회)
# 가상환경 활성화 상태에서 실행
python embeddings/build_vector_db.py

# 가상환경 활성화(window)
ai-env\Scripts\activate

# 서버 실행
uvicorn chat_server:app --reload --port 8000

# swagger
http://localhost:8000/docs