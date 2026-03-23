from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# embedding modelini başlat
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Mevcut index'i yükle
vectordb = FAISS.load_local(
    "raq_vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

# Eklemek istediğin yeni soru-cevap çiftleri
yeni_belgeler = [
    Document(
        page_content="Soru: Nakit ödeme yapabilir miyim? Kapıda ödeme var mı?\nCevap: Evet, kapıda ödeme seçeneğimiz mevcuttur, nakit olarak ödeme yapabilirsiniz.",
        metadata={"kaynak": "ek_faq", "konu": "ödeme"}
    ),
    Document(
        page_content="Soru: Depodan teslim alabilir miyim? Şubeden ürün alınır mı?\nCevap: Şu an için depodan/şubeden teslim seçeneğimiz bulunmamaktadır, tüm siparişler kargo ile gönderilmektedir.",
        metadata={"kaynak": "ek_faq", "konu": "teslimat"}
    ),
    # İstediğin kadar ekleyebilirsin...
]

# Yeni belgeleri mevcut index'e ekle
vectordb.add_documents(yeni_belgeler)

# Güncellenmiş index'i kaydet (üzerine yazar)
vectordb.save_local("raq_vectorstore")

print("✅ Yeni belgeler başarıyla eklendi!")