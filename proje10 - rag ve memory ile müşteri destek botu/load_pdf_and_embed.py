from langchain_openai import OpenAIEmbeddings # langchainin openai tabanlı vektör temsili modeli
from langchain_community.vectorstores import FAISS # vektör database
from langchain_community.document_loaders import PyPDFLoader # pdf dosyasından metin çıkarma işlemi için gerekli
from langchain_text_splitters import RecursiveCharacterTextSplitter # metni daha küçük parçalara bölme

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# os.environ["OPENAI_API_KEY"] = api_key

loader = PyPDFLoader("musteri_destek_faq.pdf")

# langchain documents objesi oluştur,
documents=loader.load()

 
# metinleri parçalamak için
# splitter, metni anlamlı parçalara ayırırken cümle veya paragraf bütünlüğünü korumaya çalışıyor
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # En uzun madde ~210 karakter, 300 güvenli üst limit
    chunk_overlap=0,     # FAQ maddeleri birbirinden bağımsız, overlap gereksiz
    separators=[
        "\n\n",          # Maddeler arası boş satır
        "\n",            
        ". Soru:",       # "1. Soru:", "2. Soru:" gibi başlıklar
    ]
)

"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # her chunk maksimum 500 karakter içerecek
    chunk_overlap=50 # her chunk bir öncekinden 50 karakter alarak örtüşebilir, bağlamı korumak için
    )
"""

# chunkları oluştur
docs = text_splitter.split_documents(documents)

# openai embedding modeli türkçe desteği var
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# chunklara ayrılmış metni vektör haline getirir ve index oluşturur
vectordb=FAISS.from_documents(documents=docs, embedding=embedding)

# oluşturulan vector veritabanını local diske kaydet
vectordb.save_local("raq_vectorstore")

print(f"Embedding ve vektör veritabanı oluşturuldu.")


