"""
web üzerinde çalışan chatbot ekranı geliştirme
streamlit framework
"""

import streamlit as st # streamlit ile web arayüzü oluşturma kütüphanesi
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# başlık ve açıklamalar
st.set_page_config(page_title="Akıllı Turizm Rehberi", page_icon="🌍")
st.title("Akıllı Turizm Rehberi")
st.markdown("Türkiye'nin dörtbir yanındaki turistik yerler hakkında bilgi almak için sorular sorabilirsiniz")

# session state(streamlit de kullanıcı geçmişini tutmak için)
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# llama model
llm = ChatOllama(model="llama3.2:3b") # qwen3:8b hem Türkçe destekli hemde daha güncel ama çok ram istiyor, llama3.2:3b daha hızlı ama Türkçe'de problemli


# mesaj kutusu: kullanıcıdan gelen mesaj
user_input = st.chat_input("Bir şehir, mekan, yemek yada aktivite sorabilirsiniz...")

if user_input:
    # yeni gelen kullanıcı mesajını ilk olarak memory'e ekliyoruz
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # tüm konuşmayı modele verecek şekilde mesajları oluşturalım
    messages = [
        SystemMessage(content="Sen bir akıllı bir turizm rehberisin. "
                      "Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, "
                      "ulaşım ve tatil önerileri hakkında yardımcı ol. "
                      "Samimi ve bilgili bir şekilde yanıt ver."),
        HumanMessage(content=user_input)
    ]  
 
    # geçmiş mesajları ekle
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # modelden yanıt al
    with st.spinner("Düşünüyorum..."):
        response = llm.invoke(messages)
    
    # asistan yanıtını hafızaya ekle
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    
# sohbet geçmişini göster
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])