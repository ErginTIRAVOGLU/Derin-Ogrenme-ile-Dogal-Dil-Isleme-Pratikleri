import re
import os
import shutil
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
import pickle

# ENV
load_dotenv()

PDF_PATH = "musteri_destek_faq.pdf"
DB_PATH = "rag_vectorstore_gelistirilmis"
BM25_PATH = "bm25.pkl"

# 1. PDF LOAD
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

full_text = " ".join([p.page_content for p in pages])

# 2. TEXT NORMALIZE
full_text = re.sub(r"\s+", " ", full_text)

# 3. FAQ PARSE (EN SAĞLAM YÖNTEM)
pattern = r'(\d{1,2}\.\s*Soru:.*?Cevap:.*?)(?=\d{1,2}\.\s*Soru:|$)'
maddeler = re.findall(pattern, full_text)

print(f"✅ Toplam FAQ: {len(maddeler)}")

# 4. CHUNK + METADATA
chunks = []

for i, madde in enumerate(maddeler):
    soru_match = re.search(r"Soru:\s*(.*?)\s*Cevap:", madde)
    cevap_match = re.search(r"Cevap:\s*(.*)", madde)

    soru = soru_match.group(1).strip() if soru_match else ""
    cevap = cevap_match.group(1).strip() if cevap_match else ""

    chunks.append(
        Document(
            page_content=f"Soru: {soru}\nCevap: {cevap}",
            metadata={
                "madde_no": i + 1,
                "soru": soru,
                "cevap": cevap
            }
        )
    )

# 5. FAISS RESET
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("🗑️ Eski vectorstore silindi")

# 6. EMBEDDING
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectordb = FAISS.from_documents(chunks, embedding)
vectordb.save_local(DB_PATH)

print("✅ FAISS index oluşturuldu")

# 7. BM25
corpus = [doc.page_content for doc in chunks]
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

with open(BM25_PATH, "wb") as f:
    pickle.dump((bm25, chunks), f)

print("✅ BM25 kaydedildi")