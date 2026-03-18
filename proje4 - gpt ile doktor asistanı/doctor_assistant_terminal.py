"""
problem tanımı: kullanıcının sağlık ile ilgili sorularını anlayan ve yanıtlayan bir gpt tabanlı doktor asistanı chatbot'u
    - kullanıcının "yaşını" ve "adını" dikkate alan cevaplar üretsin
    - mesaj geçmişini hatırlayarak diyaloğu ona göre sürdürmeli "memory"
    - langchain ve openai gpt
    - ilk olarak console'da çalışacak, sonra ise fastapi kullanarak bir web servisi oluşturulacak
    - client tarafını yazıp test edelim

veri seti: veri seti yok, onun yerine hazır gpt modelini kullanarak prompt ayarlaması yapılacak

model tanıtımı: GPT (Generative Pre-Trained Transformer) ile oluşturulan modeli kullan gpt_3.5_turbo ???
    - API üzerinden iletişim kurarak gerçek zamanlı sağlık önerilerini alalım.

Langchain: llm kütüphanesi
    - prompt yönetimi
    - memory
    - tool entegrasyonu: ai agentsler için tool kullanınmı
    - chain yapısı

API tanımalama: gpt-4.1-nano

plan/program


install libraries
    - fastapi: web api geliştirmek için bir framework (asenkron)
    - uvicorn: fastapi çalıştırmak için gereken sunucu
    - langchain
    - openai
    - python-dotenv: .env dosyasından api anahtarını almak için
    - langchain-community
    - pip install langchain-openai
    
import libraries


"""

# import libraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# ortam değişkenlerini tanımla (openai api key tanımla)
load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

# LLM + memory
# Büyük dil modeli
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    api_key=api_key
)

# memory
# --- Bellek (Memory) Yönetimi ---
# Oturum bazlı geçmiş tutmak için basit bir depot dict.
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Prompt Tasarımı ---
# Prompt şablonunu oluşturuyoruz. 
# MessagesPlaceholder, geçmiş mesajların (bellek) nereye yerleşeceğini belirtir.
prompt = ChatPromptTemplate.from_messages([
    ("system",  "Sen bir doktor asistanısın. Hasta adı {name}, {age} yaşında. "
                "Sağlık sorunları hakkında konuşmak istiyor. "
                "Yaşına uygun, dikkatli ve nazik tavsiyeler ver, ismiyle hitap et."),
    MessagesPlaceholder(variable_name="history"), # Burası bellek için ayrılıyor
    ("human", "{input}") # Kullanıcının anlık girdisi
])

# --- Zincir (Chain) Oluşturma ---
# LCEL Sözdizimi: Prompt -> LLM
chain = prompt | llm

# Zinciri Bellek ile Sarma (Wrapping)
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


# kullanıcı bilgilerini al isim ve yaş
name= input("Adınız: ")
age= input("Yaşınız: ")

# --- Yardımcı Fonksiyon: Geçmişi Yazdır ---
def print_history(session_id: str):
    """İlgili oturuma ait tüm konuşma geçmişini ekrana yazdırır."""
    history = get_session_history(session_id)
    
    print("\n" + "="*40)
    print(f"KONUŞMA GEÇMİŞİ (Toplam {len(history.messages)} mesaj)")
    print("="*40)
    
    if not history.messages:
        print("Henüz bir konuşma yapılmadı.")
    else:
        for msg in history.messages:
            # Mesaj tipine göre (Human/AI) kimin konuştuğunu belirle
            role = "Siz" if msg.type == "human" else "Doktor Asistanı"
            print(f"[{role}]: {msg.content}")
            print("-" * 20)
    print("="*40 + "\n")

print(f"\nTeşekkürler {name} ({age}). Size nasıl yardımcı olabilirim? (Çıkmak için 'exit' yazın)\n")
print("(Komutlar: 'exit' = Çıkış, 'gecmis' = Konuşma Geçmişi)\n")

# chatbot döngüsü tanımlama
session_id = "terminal_user_1" # Sabit bir oturum ID'si

while True:
    user_input = input("Siz: ")
    
    if user_input.lower() in ["exit", "quit", "çıkış"]:
        print("Doktor Asistanı: Geçmiş olsun, iyi günler dilerim!")
        break
    # Geçmiş yazdırma kontrolü
    if user_input.lower() in ["gecmis", "history", "geçmiş"]:
        print_history(session_id)
        continue # Döngünün başına dön, bot cevap vermesin
        
    # invoke sırasında config içinde session_id belirtmeliyiz.
    # Bu sayede o kullanıcının geçmişi yüklenir.
    response = chain_with_history.invoke(
        {"input": user_input, "name": name, "age": age},
        config={"configurable": {"session_id": session_id}}
    )
    
    print(f"Doktor Asistanı: {response.content}")




