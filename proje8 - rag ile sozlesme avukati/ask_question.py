"""
problem tanımı: sözleşme asistanı 
    - kullanıcının yüklediği bir sözleşme dosyasından içerik çıkarmak
    - bu içeriği vektörel olarak temsil edelim yani (embedding)
    - faiss kullanarka hızlı arama yapabilen bir vektöre veri tabanı oluştur.
    - kullanıcı sorularını al, sonra git db'den bilgiyi getir, sonra gpt-4.1-nano ile cevapla

kullanılan teknolojiler:
    - embedding: metni vektörleştirme
    - faiss: hızlı benzerlik araması için vektöre veri tabanı
    - gpt: metin üretimi ve cevaplama    

RAG: Retrieval Augmented Generation: dil bmodellerine bilgi desteği sağlayan bir teknik
    - kullanıcı sorularını al, ilgili bilgiliyi veritabanından getir, sonra gpt ile cevapla
    - kullanıcı sorusu embedding'e (vektörleştirme) dönüştürülür, faiss(db) üzerinden en alakalı içerik (chunk) getiriliyo
    - augmentation: bulunan metin parçaları (chunks) llm'in anlayabileceği bir formata dönüştürülüyo
    - generation: dil modeli bu bilgiler ile mantıklı yanıt üretir
        - 1) tarih
        - 2) ücret
        - 3) taraflar -> Ucanble Tenkoloji - Kaan Can Yılmaz

plan/program
    - Sözleşme belgesinin hazırlanması
    - metin çıkarma ve parçalama
    - embedding ve faiss ile vector db oluşturma
    - soru cevap sistemi

install libraries: freeze

"""


import os 
import pickle
from urllib import response
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from openai import OpenAI

# .env dosyasını oku
load_dotenv()

# llm kurulumu
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) # openai istemcisi

model = SentenceTransformer("all-MiniLM-L6-v2") # embedding model

# faiss index dosyasını yükle (önceden yüklenmiş vektör veritabani)
index = faiss.read_index("data/AG_Application_Development_Contract_index.faiss")

# chunklanmis metin verisini yükle
with open("data/AG_Application_Development_Contract_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
    
# chat başlat
while True:
    query = input("Ask the question: ")
    
    # çıkış komutu
    if query == "exit":
        print("Exiting...")
        break
    
    # soruyu vektöre çevir (embedding)
    query_embedding = model.encode([query]) # soruyu embedding'e dönüştür
    
    # fais veri tabanından en yalın 3 chunk i ara -> 3 hyper parametre
    k = 3 # en yakın 3 chunk
    distances, indices = index.search(np.array(query_embedding), k)
    
    # bulunan chunkları birleştirerek bağlam yani context oluştur
    retrieved_chunks = [chunks[i] for i in indices[0]] # ilk satırdaki chunklar
    context = "\n ---- \n".join(retrieved_chunks)
    
    #llm'e gönderilecek
    prompt = f"""
            You are a contract lawyer AI assistant. Based on the contract context below, 
            answer the user's question clearly.
            
            Context:
            {context}
            
            Question:
            {query}
            
            Answer:                       
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8 # daha kararlı cevaplar için düşük değer
    )
    
    print("AI Assistant. \n",response.choices[0].message.content.strip())
            