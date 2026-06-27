import os
import requests
import fitz
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# 1. 환경설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

load_dotenv(PROJECT_ROOT / ".env")
NONGSARO_API_KEY = os.getenv("NONGSARO_API_KEY")

docs_dir = str(EMBEDDINGS_DIR / "docs")
downloads_dir = str(EMBEDDINGS_DIR / "downloads")
os.makedirs(downloads_dir, exist_ok=True)

# 2. 기존 PDF 로드 (특정 페이지 범위만)
all_docs = []
for path in glob(os.path.join(docs_dir, "*.pdf")):
    filename = os.path.basename(path)
    loader = PyPDFLoader(path)
    pages = loader.load()
    
    # docs에 있는 딸기,참외.pdf 사용
    for page in pages:
        page.metadata["source"] = filename
        all_docs.append(page)
    
    print(f"📄 {filename}: {len(pages)}페이지 로드 완료")
    

# 3. 농진청 병해충 PDF → Document 변환
def download_file(url, filename):
    res = requests.get(url)
    if res.status_code == 200:
        path = os.path.join(downloads_dir, filename)
        with open(path, "wb") as f:
            f.write(res.content)
        return path
    return None

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def get_nongsaro_documents(keywords, year="2024"):
    docs = []
    url = "http://api.nongsaro.go.kr/service/dbyhsCccrrncInfo/dbyhsCccrrncInfoList"
    params = {"apiKey": NONGSARO_API_KEY, "sYear": year}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return docs
    root = ET.fromstring(res.content)
    items = root.findall(".//item")
    for item in items:
        title = item.findtext("cntntsSj")
        file_url = item.findtext("downFile")
        file_name = item.findtext("rtnOrginlFileNm")
        if file_url and file_name:
            path = download_file(file_url, file_name)
            if path:
                text = extract_text_from_pdf(path)
                for kw in keywords:
                    if kw in text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": f"NongSaRo_{file_name}", "matched_keyword": kw}
                        ))
                        break
    return docs

# 4. 병해충 PDF 로드 및 병합
keywords = ["딸기", "참외", "단호박"]
nongsaro_docs = get_nongsaro_documents(keywords)
all_docs += nongsaro_docs

# 5. 문서 청크 및 벡터화
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

# HuggingFace 임베딩 사용
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

batch_size = 50
all_embs = []
for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    embs = embeddings.embed_documents(batch)
    all_embs.extend(embs)

# 6. FAISS 저장
vectorstore = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, all_embs)),
    embedding=embeddings,
    metadatas=metadatas
)
index_path = str(EMBEDDINGS_DIR / "faiss_index")
os.makedirs(index_path, exist_ok=True)
vectorstore.save_local(index_path)

print(f"✅ 총 {len(chunks)}개 문서를 인덱싱하여 저장 완료: {index_path}")
