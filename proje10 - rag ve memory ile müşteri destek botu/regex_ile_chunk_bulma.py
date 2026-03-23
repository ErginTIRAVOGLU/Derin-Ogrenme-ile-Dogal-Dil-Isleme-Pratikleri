import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os, shutil

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("musteri_destek_faq.pdf")
pages = loader.load()
full_text = " ".join([p.page_content for p in pages])

# ✅ Negatif lookbehind ile düzeltildi
maddeler = re.split(r'(?<!\d)(?=\d{1,2}\.\s*Soru:)', full_text)

# Kısa/boş chunk'ları filtrele
maddeler = [m.strip() for m in maddeler if len(m.strip()) > 10]

print(f"✅ Toplam FAQ maddesi: {len(maddeler)}")  # 25 olmalı!
for i, madde in enumerate(maddeler):
    print(f"\n[{i+1}] {len(madde)} karakter: {madde[:80]}...")

# Doğrulama kontrolü
if len(maddeler) != 25:
    print(f"\n⚠️ UYARI: {len(maddeler)} chunk var, 25 olmalıydı!")
else:
    print("\n🎉 Chunk sayısı doğru: 25")

chunks = [
    Document(
        page_content=madde,
        metadata={"kaynak": "musteri_destek_faq.pdf", "madde_no": i+1}
    )
    for i, madde in enumerate(maddeler)
]

if os.path.exists("raq_vectorstore"):
    shutil.rmtree("raq_vectorstore")
    print("🗑️ Eski index silindi")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vectordb = FAISS.from_documents(chunks, embedding)
vectordb.save_local("raq_vectorstore")
print("✅ Index yeniden oluşturuldu!")

# regex_ile_chunk_bulma.py sonuna ekleyin
from langchain_core.documents import Document

yeni_belgeler = [
    Document(
        page_content="Soru: Nakit ödeme yapabilir miyim? Kapıda ödeme var mı?\nCevap: Evet, kapıda ödeme seçeneğimiz mevcuttur, nakit olarak ödeme yapabilirsiniz.",
        metadata={"kaynak": "ek_faq", "madde_no": 26}
    ),
    Document(
        page_content="Soru: Depodan teslim alabilir miyim? Şubeden ürün alınır mı?\nCevap: Şu an için depodan/şubeden teslim seçeneğimiz bulunmamaktadır.",
        metadata={"kaynak": "ek_faq", "madde_no": 27}
    ),
]

vectordb.add_documents(yeni_belgeler)
vectordb.save_local("raq_vectorstore")
print(f"✅ Manuel belgeler eklendi! Toplam: {vectordb.index.ntotal} chunk")