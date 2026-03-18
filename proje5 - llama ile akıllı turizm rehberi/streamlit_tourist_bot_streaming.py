"""
web üzerinde çalışan chatbot ekranı geliştirme stream özelliği ile
streamlit framework
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

# streamlit için özel streaming callback tanımı
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.final_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.final_text += token
        self.placeholder.markdown(self.final_text + "▌")  # imleç efekti

# başlık ve açıklamalar
st.set_page_config(page_title="Akıllı Turizm Rehberi (Streaming)", page_icon="🌍")
st.title("🌍 Akıllı Turizm Rehberi (Streaming)")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi almak için sorular sorabilirsiniz")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# sohbet geçmişini göster
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# mesaj kutusu
user_input = st.chat_input("Bir şehir, mekan, yemek veya aktivite sorabilirsiniz...")

if user_input:
    # kullanıcı mesajını hafızaya ekle
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # kullanıcı mesajını göster
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # asistan yanıtı için placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_handler = StreamHandler(placeholder)
        
        # llama model - streaming aktif
        llm = ChatOllama(
            model="llama3.2:3b",
            streaming=True,
            callbacks=[stream_handler]
        )
        
        # mesajları hazırla
        messages = [
            SystemMessage(content="Sen bir akıllı turizm rehberisin. "
                          "Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, "
                          "ulaşım ve tatil önerileri hakkında yardımcı ol. "
                          "Samimi ve bilgili bir şekilde yanıt ver.")
        ]
        
        # geçmiş mesajları ekle (son mesaj hariç çünkü zaten user_input var)
        for msg in st.session_state.messages[:-1]:  # ← son mesajı atla!
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # şimdiki kullanıcı mesajını ekle
        messages.append(HumanMessage(content=user_input))
        
        # modelden yanıt al (streaming)
        response = llm.invoke(messages)
        
        # final metni temizle (imleç kaldır)
        placeholder.markdown(response.content)
    
    # asistan yanıtını hafızaya ekle
    st.session_state.messages.append({"role": "assistant", "content": response.content})