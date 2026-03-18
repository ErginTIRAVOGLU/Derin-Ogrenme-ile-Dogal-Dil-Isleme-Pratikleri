"""
problem tanımı: kullanıcılar yazılı olarak soru soracak, gerçek zamanlı ve doğan şekilde yanıtlar alabilecek
    - akıllı turizm rehberi
    - Türkiye özelinde: tarihi yerler, kültürel etkinlikler, yemekler, ulaşım ...
    - llama 3.2 3B parametreli modeli ile cevapları streamlit üzerinden gerçek zamanlı görselleştireceğiz
    
model tanıtımı: LLAMA (Large Language Model Meta AI) 3.2 3B
    - açık kaynak: akademik ve ticari kullanımlar için uygundur
    - verimli: daha az parametre ile aynı performansı sergiliyor
    - moduler: 1B, 3B, 8B, 70B parametreye sahip modelleri var
    - local'de çalışabilir 

Plan/Program:

install libraries: freeze

ollama indir ve llama kur

import libraries



"""

# import libraries

# import libraries
from typing import Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# llama model
llm = ChatOllama(model="qwen3:8b") # qwen3:8b hem Türkçe destekli hemde daha güncel ama çok ram istiyor, llama3.2:3b daha hızlı ama Türkçe'de problemli

# prompt template - history placeholder'ı önemli!
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir akıllı turizm rehberisin. "
               "Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, "
               "ulaşım ve tatil önerileri hakkında yardımcı ol. "
               "Samimi ve bilgili bir şekilde yanıt ver."
               "Sahip olduğun bilgilerden bazıları, örneğin transfer ücretleri, otel ücretleri, yemek ücretleri gibi, "
               "güncel bilgiler değil, çok eski bilgiler, o yüzden bu fiyat bilgilerini paylaşma."),
    MessagesPlaceholder(variable_name="history"),  # Geçmiş mesajlar buraya gelir
    ("human", "{input}")
])

# chain oluştur
chain = prompt | llm

# memory - konuşma geçmişi takip etme
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print(f"Yeni session oluşturuluyor: {session_id}")
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# history ile chain'i sar
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# welcome message
print("=" * 50)
print("🏰 Akıllı Turizm Rehberine Hoş Geldiniz!")
print("=" * 50)
print("Size Türkiye'deki şehirler, tarihi yerler, yöresel yemekler,")
print("ulaşım ve tatil önerileri konusunda yardımcı olabilirim.")
print("Çıkmak için 'exit' yazabilirsiniz.\n")

# terminal üzerinden konuşma
while True:
    user_input = input("👤 Soru: ")
    
    if user_input.lower() == "exit":
        print("\n👋 Görüşmek üzere! İyi yolculuklar!")
        break
    
    if not user_input.strip():
        continue
    
    # modelden yanıt al - invoke ile çağır!
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "turizm_user_001"}}
    )
    
    print(f"🤖 Asistan: {response.content}\n")