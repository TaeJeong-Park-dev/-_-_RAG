from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from config import OPENAI_API_KEY

import re

# HTML 파일 경로 리스트
html_file_paths = [
    "/usr/workspace/processed/2018년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2019년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2020년 1학기 개설과목.html",
    "/usr/workspace/processed/2020년 2학기 개설과목.html",
    "/usr/workspace/processed/2020년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2021년 1학기 개설과목.html",
    "/usr/workspace/processed/2021년 2학기 개설과목.html",
    "/usr/workspace/processed/2021년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2022년 1학기 개설과목.html",
    "/usr/workspace/processed/2022년 2학기 개설과목.html",
    "/usr/workspace/processed/2022년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2023년 1학기 개설과목.html",
    "/usr/workspace/processed/2023년 2학기 개설과목.html",
    "/usr/workspace/processed/2023년 입학생 졸업이수학점.html",
    "/usr/workspace/processed/2024년 1학기 개설과목.html",
    "/usr/workspace/processed/2024년 2학기 개설과목.html",
    "/usr/workspace/processed/2024년 입학생 졸업이수학점.html",
]

# 문서 로드 및 메타데이터 추가
all_documents = []

for file_path in html_file_paths:
    # 파일명에서 년도 추출
    year_match = re.search(r"(\d{4})년", file_path)
    year = year_match.group(1) if year_match else "Unknown"
    
    # 문서 로드
    loader = UnstructuredHTMLLoader(file_path)
    loaded_documents = loader.load()
    
    # 문서 분할
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(loaded_documents)
    
    # 메타데이터 추가
    for doc in split_docs:
        doc.metadata["year"] = year
        all_documents.append(doc)

print(f"총 {len(all_documents)}개의 문서가 준비되었습니다.")

# OpenAI 임베딩 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)

# FAISS 인덱스 생성 및 추가
vectorstore = FAISS.from_documents(all_documents, embedding=embeddings)

# 리트리버 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"filter": lambda doc: doc.metadata["year"] == "2023"})

print("FAISS 리트리버가 초기화되었습니다.")
