# GardenDoctor AI Server

AI 기반 작물 질병 이미지 분석을 통한 도시농업 지원 시스템, “텃밭닥터”의 FastAPI 추론 서버입니다.
사용자가 업로드한 작물 이미지를 검증한 뒤 작물 분류와 작물별 질병 진단을 수행하고, 진단 결과와 피드백/챗봇 대화 이력을 SQLite에 저장합니다.

## 주요 기능

- 이미지 업로드 기반 작물 질병 진단
- YOLO 기반 작물 이미지 유효성 검증
- 1단계 작물 분류, 2단계 작물별 질병 분류
- PyTorch 기반 모델 서빙
- 진단 결과 및 사용자 피드백 저장
- LangGraph/ReAct 기반 농업 상담 챗봇
- PDF/농사로 데이터 기반 FAISS 벡터 DB 구축
- 피드백 기반 재학습 데이터셋 구성 스크립트

## 폴더 구조

```text
.
├── ai_server/
│   ├── api/
│   │   └── main.py                 # FastAPI 앱과 API 라우트
│   ├── db/
│   │   └── database.py             # SQLAlchemy 모델, SQLite 세션
│   ├── inference/
│   │   ├── predict.py              # 작물 분류 및 질병 진단 파이프라인
│   │   ├── validation.py           # YOLO 작물 이미지 유효성 검증
│   │   └── disease_catalog.py      # 질병 상세 설명/처방 데이터
│   ├── chatbot/
│   │   └── agent.py                # LangGraph/ReAct 챗봇 에이전트
│   ├── rag/
│   │   └── build_vector_db.py      # PDF/API 문서 임베딩 및 FAISS 인덱스 생성
│   ├── scripts/
│   │   └── retrain_data_builder.py # 피드백 기반 재학습 데이터 구성
│   └── schemas.py                  # Pydantic 요청/응답 스키마
├── ai_models/                      # 모델 가중치
├── embeddings/
│   ├── docs/                       # RAG 원본 PDF
│   ├── downloads/                  # 농사로 다운로드 문서
│   └── faiss_index/                # FAISS 벡터 DB
├── test_data/                      # 샘플 이미지
├── docs/                           # 프로젝트 보고서
├── chat_server.py                  # 기존 실행 호환용 wrapper
├── Dockerfile
└── requirements.txt
```

## 실행

```bash
python -m venv ai-env
source ai-env/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
ai-env\Scripts\activate
pip install -r requirements.txt
```

벡터 DB 최초 생성:

```bash
python -m ai_server.rag.build_vector_db
```

서버 실행:

```bash
uvicorn ai_server.api.main:app --reload --port 8000
```

기존 명령도 wrapper를 통해 동작합니다.

```bash
uvicorn chat_server:app --reload --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Docker

```bash
docker build -t gardendoctor-api:1.0 .
docker run -d -p 8000:8000 gardendoctor-api:1.0
```

## 환경 변수

`.env` 파일에 외부 API 키를 설정합니다.

```env
OPENAI_API_KEY=
TAVILY_API_KEY=
NONGSARO_API_KEY=
```

## 주요 API

- `GET /`: 서버 상태 확인
- `POST /diagnose`: 이미지 파일 업로드 진단
- `POST /diagnose-by-url`: 이미지 URL 진단
- `POST /diagnoses/{diagnosis_id}/feedback`: 진단 결과 피드백 저장
- `POST /api/chat`: 챗봇 질문 전송
- `GET /api/chat/sessions`: 챗봇 세션 목록
- `GET /api/chat/sessions/{session_id}`: 특정 세션 메시지 조회
- `DELETE /api/chat/sessions/{session_id}`: 세션 삭제
