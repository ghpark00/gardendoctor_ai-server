# GardenDoctor AI Server

An AI-powered crop disease diagnosis server for the **GardenDoctor (텃밭닥터)** project, designed to support urban farming users by analyzing crop images, predicting plant diseases, and providing diagnosis results through a FastAPI-based serving pipeline.

This repository contains the AI server component of GardenDoctor, including image preprocessing, crop disease inference, diagnosis result handling, chatbot/RAG modules, and lightweight database integration for diagnosis and feedback records.

---

## 1. Project Overview

**GardenDoctor** is an integrated urban agriculture support application that helps beginner urban farmers manage crops more effectively.
<img width="184" height="398" alt="image" src="https://github.com/user-attachments/assets/c3a727a1-dd3c-481b-a951-c68129039fbf" />


Urban farming users often struggle with limited farming knowledge, delayed expert consultation, and difficulty responding to crop diseases or pests. GardenDoctor addresses these issues by providing:

* AI-based crop disease diagnosis from uploaded images
* Real-time diagnosis result delivery
* 24-hour agricultural chatbot consultation
* Crop diary and user crop management
* Location-based farm information search

The AI server focuses on the image diagnosis and AI consultation layer of the system. Users can upload crop images through the mobile application, and the AI server analyzes the image, validates whether it is suitable for diagnosis, predicts the disease class, and returns structured diagnosis results to the backend service.

---

## 2. My Role

**Role:** Team Leader / AI Engineer
**Period:** 2025.07 ~ 2025.08
**Project:** K-Software Empowerment Bootcamp

I designed and implemented the AI pipeline and serving structure for the crop disease diagnosis service.

My main responsibilities included:

* Designing the AI inference pipeline for crop image diagnosis
* Developing PyTorch-based image classification models
* Improving disease diagnosis performance using ConvNeXt-Tiny
* Building a FastAPI-based AI serving environment
* Implementing the flow from image upload to real-time inference and result response
* Converting trained AI models into a service-ready inference module
* Managing the overall AI development direction and integration with backend services

---

## 3. Key Features

### AI Crop Disease Diagnosis

The server analyzes uploaded crop images and predicts disease classes using a PyTorch-based image classification model.

The diagnosis pipeline includes:

1. Image upload request from the client/backend
2. Image preprocessing and tensor transformation
3. Crop/disease inference using a trained CNN model
4. Confidence score calculation
5. Diagnosis result formatting
6. Response delivery to the backend service

---

### Crop Validity Verification

To prevent unnecessary inference on invalid images, the system includes a crop validity verification step.

This step filters out irrelevant images such as people, objects, landscapes, or non-crop photos before running disease diagnosis. This improves system reliability and reduces unnecessary model computation.

---

### ConvNeXt-Tiny Based Classification

The project uses **ConvNeXt-Tiny** as the main image classification model for crop disease diagnosis.

ConvNeXt-Tiny was selected because it provides a strong balance between image classification performance and inference efficiency, making it suitable for a practical AI serving environment.

---

### FastAPI-Based AI Serving

The AI model is served through a FastAPI application, allowing the backend server and client application to request diagnosis results through RESTful APIs.

FastAPI was used because it is lightweight, fast, and suitable for deploying machine learning inference services.

---

### Chatbot and RAG Module

The project includes a chatbot module designed to support agricultural Q&A after diagnosis.

The chatbot architecture uses:

* LangGraph for conversation flow control
* FAISS for vector-based agricultural knowledge retrieval
* Hugging Face sentence-transformer embeddings
* LLM API integration for natural language response generation

This allows the service to provide follow-up explanations and management guidance after the AI diagnosis result.

---

## 4. System Architecture

The overall GardenDoctor system consists of the following layers:

```text
Client Layer
├── Flutter Mobile App
│   ├── Crop disease diagnosis
│   ├── Crop diary
│   ├── Chatbot consultation
│   └── Farm information search
│
Backend Layer
├── Spring Boot Backend
│   ├── User authentication
│   ├── Crop and diary APIs
│   ├── Farm information APIs
│   └── FastAPI AI server integration
│
AI Server Layer
├── FastAPI AI Server
│   ├── Image preprocessing
│   ├── Crop validity verification
│   ├── Disease classification inference
│   ├── Diagnosis result response
│   └── Chatbot/RAG module
│
Data Layer
├── MySQL
├── SQLite
├── Redis
├── FAISS Vector DB
└── AWS S3
```

---

## 5. AI Pipeline

```text
Image Upload
    ↓
Image Validation
    ↓
Preprocessing
    ↓
PyTorch Model Inference
    ↓
Disease Class Prediction
    ↓
Confidence Score Calculation
    ↓
Diagnosis Result Formatting
    ↓
Response to Backend / Client
```

The AI pipeline was designed to make the trained model usable in an actual service environment, not only as an experimental notebook model.

---

## 6. Repository Layout

```text
gardendoctor_ai-server/
├── ai_server/
│   ├── api/
│   │   └── main.py                  # FastAPI application entry point
│   │
│   ├── chatbot/
│   │   └── agent.py                 # LangGraph-based chatbot agent
│   │
│   ├── db/
│   │   └── database.py              # Lightweight DB connection and setup
│   │
│   ├── inference/
│   │   ├── disease_catalog.py       # Disease label and catalog configuration
│   │   ├── predict.py               # Image preprocessing and model inference
│   │   └── validation.py            # Crop validity verification logic
│   │
│   ├── rag/
│   │   └── build_vector_db.py       # FAISS vector DB construction
│   │
│   ├── scripts/
│   │   └── retrain_data_builder.py  # Dataset/retraining utility script
│   │
│   └── schemas.py                   # Request/response schemas
│
├── config/                          # Configuration files
├── test_data/                       # Sample test data
├── Dockerfile                       # Docker image definition
├── requirements.txt                 # Runtime dependencies
├── requirements.dev.txt             # Development dependencies
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 7. Tech Stack

### AI / Machine Learning

* Python
* PyTorch
* ConvNeXt-Tiny
* CNN-based image classification
* Image preprocessing and augmentation
* Hugging Face sentence-transformers

### AI Serving

* FastAPI
* Uvicorn
* Pydantic
* SQLite for lightweight diagnosis and chat history storage

### Chatbot / RAG

* LangGraph
* FAISS Vector DB
* OpenAI API
* Agricultural knowledge retrieval pipeline

### Infrastructure / Deployment

* Docker
* AWS S3
* Redis
* GitHub

---

## 8. Dataset

The project used public agricultural image datasets for crop disease diagnosis.

Main dataset source:

* AI Hub crop disease diagnosis image dataset

The dataset was used for training, validation, and testing of crop disease classification models. Image preprocessing, augmentation, and class distribution checks were applied to improve model robustness.

---

## 9. Model Weights

Trained model weight files are excluded from this repository because of file size limitations and repository management purposes.

The repository focuses on:

* AI server source code
* Inference pipeline
* Model loading logic
* API serving structure
* Chatbot/RAG modules
* Docker-based execution environment

Model checkpoint files such as `.pth`, `.pt`, `.ckpt`, and `.onnx` are ignored by Git.

---

## 10. Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/ghpark00/gardendoctor_ai-server.git
cd gardendoctor_ai-server
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Server

```bash
uvicorn ai_server.api.main:app --host 0.0.0.0 --port 8000 --reload
```

After running the server, API documentation is available at:

```text
http://localhost:8000/docs
```

---

## 11. Docker Usage

Build the Docker image:

```bash
docker build -t gardendoctor-ai-server .
```

Run the container:

```bash
docker run -p 8000:8000 gardendoctor-ai-server
```

---

## 12. Example API Flow

```text
POST /diagnosis
Content-Type: multipart/form-data

Input:
- crop image file

Output:
- predicted crop/disease class
- confidence score
- diagnosis message
- optional follow-up guide
```

---

## 13. Project Outcome

Through this project, the AI model was converted from a standalone training artifact into a deployable service component.

The final AI server provides:

* A structured FastAPI serving layer
* Image-based disease diagnosis logic
* Crop validity verification
* Chatbot/RAG-based agricultural consultation support
* A maintainable project structure for future model improvement and service integration

---

## 14. Related Repositories

GardenDoctor consists of multiple repositories:

* Mobile App
* Backend Server
* AI Server

This repository is responsible for the AI server and inference pipeline.

---

## 15. Contributor

**Kwanho Park**
Team Leader / AI Engineer

Responsible for AI pipeline design, PyTorch model development, ConvNeXt-Tiny based disease diagnosis, FastAPI serving implementation, and AI service integration.
