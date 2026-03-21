"""
    Vector Data Builder
    Faiss, yoğun vektörlerin verimli benzerlik araması ve kümelenmesi için kullanılan bir kütüphanedir.

"""

import os
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer # embedding
import faiss # vector db
import numpy as np
import pickle # for saving the vector database

# program için olarak dosya olarak .pdf yükleyelim
# .pdf 'ten metin dönüşümü yapmamız lazım

def extract_text_from_pdf(pdf_path): # pdf dosyasindan metin çıkartma
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    return text

# print(extract_text_from_pdf("./data/us_clean_contract.pdf"))


# uzun metni daha küçük parçalara böl
def chunk_text(text, chunk_size=500): # metni belirtilen karakter uzunluğuna göre böl
    chunks=[]
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < chunk_size:
            current += " " + line.strip()
        else:
            chunks.append(current.strip())
            current = line.strip()
    if current:
        chunks.append(current.strip())
    return chunks

# print(chunk_text(extract_text_from_pdf("./data/us_clean_contract.pdf"),500))
# print(chunk_text(extract_text_from_pdf("./data/AG_Application_Development_Contract.pdf"),500))
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# https://huggingface.co/BAAI/bge-base-en-v1.5 (ARAŞTIR)
model = SentenceTransformer("all-MiniLM-L6-v2")

# pdf yolunu belirt
pdf_path = "./data/AG_Application_Development_Contract.pdf"

# pdf ten metin çıkart
text = extract_text_from_pdf(pdf_path)

# metni chunklara bölelim
chunks = chunk_text(text, chunk_size=500)

# her chunk için embeddin (vektörel temsil) oluşturalım
embeddings = model.encode(chunks)

print(f"Embedding shape: {embeddings.shape}") # (n_chunks, embedding_dim) -> number of chunks ve embedding dimens
# faiss index oluştur
dimension = embeddings.shape[1] # embedding (vector) boyutu
index = faiss.IndexFlatL2(dimension) # L2 Norm (Euclidean distance) kullanarak benzerlik arama
index.add(np.array(embeddings)) # embeddingleri indexe ekle

# faiss index'i ve chunları kaydet
faiss.write_index(index, "data/AG_Application_Development_Contract_index.faiss")
with open("data/AG_Application_Development_Contract_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index ve chunklar kayıt edildi.")



