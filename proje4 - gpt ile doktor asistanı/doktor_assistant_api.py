import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Ortam değişkenleri
load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

# FastAPI uygulaması
app = FastAPI(title="GPT Doktor Asistanı")

# --- 1. GLOBAL SEVİYEDE TANIMLAMALAR (Hafıza kaybolmaması için) ---
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print(f"UYARI: Yeni session oluşturuluyor -> {session_id}")
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# LLM Yapılandırması
llm = ChatOpenAI(
    model="gpt-4.1-nano",  # Düzeltildi: Geçerli model
    temperature=0.7,
    api_key=api_key
)

# --- 2. SCHEMA TANIMLARI ---
class ChatRequest(BaseModel):
    name: str
    age: int
    message: str
    # Her kullanıcı için ayrı hafıza tutmak için client'tan session_id alabiliriz,
    # ya da name'e göre kendimiz üretebiliriz. Şimdilik name'i kullanalım.

class ChatResponse(BaseModel):
    response: str

# --- 3. ZİNCİR (CHAIN) TANIMI ---
# Prompt ve Chain sadece bir kere oluşturulmalı (performans için fonksiyon dışı)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir doktor asistanısın. Hasta adı {name}, {age} yaşında. "
               "Sağlık sorunları hakkında konuşmak istiyor. "
               "Yaşına uygun, dikkatli ve nazik tavsiyeler ver, ismiyle hitap et."),
    MessagesPlaceholder(variable_name="history"), # Bellek yeri
    ("human", "{input}") # Kullanıcı girdisi
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- 4. ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    # ID oluştur
    current_session_id = f"session_{request.name}"
    
    # DEBUG: Konsola ID'yi yazdır
    print(f"Chat istek geldi. Kullanılan Session ID: {current_session_id}")

    try:
        # DİKKAT: Async fonksiyonlarda senkron 'invoke' kullanmak performansı düşürür.
        # 'ainvoke' kullanmak daha doğrudur ama 'invoke' da çalışır.
        response = chain_with_history.invoke(
            {
                "input": request.message,
                "name": request.name,
                "age": request.age
            },
            config={"configurable": {"session_id": current_session_id}}
        )
        
        # DEBUG: Hafızaya eklenip eklenmediğini kontrol edelim
        current_history = get_session_history(current_session_id)
        print(f"Hafızadaki mesaj sayısı: {len(current_history.messages)}")
        
        return ChatResponse(response=response.content)
    
    except Exception as e:
        print(f"Hata: {e}")
        return ChatResponse(response="Hata")

@app.get("/history/{session_id}")
async def get_history_endpoint(session_id: str):
    # Buraya gelen ID'nin chat sırasında kullanılan ID ile AYNI olmasına dikkat edin.
    # Chat'te "session_Ali" oluşturulduysa, buraya da "session_Ali" gelmeli.
    print(f"History isteği geldi. Aranan ID: {session_id}")
    
    if session_id not in store:
        return {"error": "Bu ID ile bir oturum bulunamadı", "session_id": session_id}

    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        messages.append({"role": msg.type, "content": msg.content})
    return {"session_id": session_id, "count": len(messages), "history": messages}

# swagger: http://127.0.0.1:8000/docs