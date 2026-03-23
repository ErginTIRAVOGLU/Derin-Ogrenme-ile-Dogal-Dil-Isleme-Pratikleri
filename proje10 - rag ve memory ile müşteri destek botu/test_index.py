# test_index.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")


embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectordb = FAISS.load_local(
    "raq_vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

# Toplam belge sayısını kontrol et
print(f"Toplam belge sayısı: {vectordb.index.ntotal}")

# Similarity score ile birlikte ara
test_sorgular = [
    "depodan teslim",
    "depodan teslim alabilir miyim",
    "nakit ödeme",
]

for sorgu in test_sorgular:
    print(f"\n{'='*50}")
    print(f"Sorgu: '{sorgu}'")
    sonuclar = vectordb.similarity_search_with_score(sorgu, k=3)
    for doc, score in sonuclar:
        print(f"Skor: {score:.4f} | İçerik: {doc.page_content[:80]}...")