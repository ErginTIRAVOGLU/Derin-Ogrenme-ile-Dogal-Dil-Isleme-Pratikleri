"""
problem tanımı: Akıllı Müşteri Destek Sistemi
    - müşterileri sık sık benzer soruları sorarlar:
        - şifremi unuttum
        - faturamı nerden alabilirim
        - iade süresi kaç gün
        - yurt dışına gönderim yapıyormusunuz
    - çözüm:
        - .pdf dosyasını ( sıkça sorulan soruları ) vektör veri tabanına dönüştürülecek
        - kullanıcıdan gelen sorular veritabanına sorgulanır ve gpt türkçe cevaplar üretir
        - 


Kullanılan teknolojiler:
    - langchain: rag mimarisi kurmak için
    - faiss: embeddingleri saklama için hızlı bir vektör veri tabanı
    - openai: soru cevap için llm
    - streamlit: web arayüzü
    
veri seti:
    - Soru: "Yurdışı satışlarınız bulunuyor mu?"
    - Cevap: "Hayır"
    
    - Soru: "Faturamı nereden alabilirim"
    - Cevap: "Faturanız 3 iş günü içerisinde teslim edilecektir"


plan/program:
    - SSS bilgileri içeren pdf
    - kullanıcı bu dosyayı arayüzden yükleyecek
    - pdf metni parçaya ayıracak ve embeddingler çıkarılacak
    - kullanıcı soru sorduğu zaman vektör db 'den benzer içerikler getirilir, gpt ile cevap oluşturulur
    - konuşma geçmişi memory ile saklanır ve sonraki yanıtlara bağlam oluşturulur.

install libraries: freeze

"""
from typing import Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # vektör database
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")


# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen yardımcı bir müşteri destek botusun.
    
KURALLAR:
1. SADECE aşağıdaki bağlamdaki bilgileri kullan.
2. Bağlamda ilgili bilgi varsa MUTLAKA kullan, kendi yorumunu katma.
3. Bağlamda bir hizmetin MEVCUT OLMADIĞI yazıyorsa, bunu açıkça belirt.
   Örnek: "Şu an için bu hizmetimiz bulunmamaktadır."
4. Bağlamda hiç ilgili bilgi yoksa: 
   "Bu konuda bilgim bulunmuyor, müşteri hizmetlerimize başvurabilirsiniz." de.
5. "Mevcut değil" ile "Bilgim yok" FARKLIDIR, doğru olanı kullan!

Bağlam: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# embedding modelini başlat (text -> vektör dönüşümü)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# daha önce oluşturulmuş vektör veritabanı yükle
vectordb = FAISS.load_local(
    "raq_vectorstore", 
    embedding,
    allow_dangerous_deserialization=True # güvenlik uyarısı bastırma
)
retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)


# llm
llm= ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0, # sıfıra ne kadar yakınsa o kadar az halüsülasyon, 1 e nekadar yakınsa o kadar halüsünasyon görür
)

# Zincir
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "history": lambda x: x["history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# konuşma geçmişsi için memory oluşturma
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print(f"Yeni session oluşturuluyor: {session_id}")
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

print(f"Müşteri destek botuna hoş geldiniz!")
# Zincir oluşturmadan önce test edin
def debug_retriever(soru):
    docs = retriever.invoke(soru)
    print(f"\n🔍 DEBUG - '{soru}' için getirilen belgeler:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    return docs

while True:
    user_input = input("Soru: ")
    if user_input.lower() == "exit":
        break
    
    debug_retriever(user_input) 
     
    response = chain_with_history.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": "kullanici_123"}},
    )
    print(f"Cevap: {response}")

 