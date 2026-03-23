import re
import os
import shutil
import streamlit as st
import pickle
import tempfile
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

# Sabitler
DB_PATH = "rag_vectorstore_gelistirilmis"
BM25_PATH = "bm25.pkl"

# Sayfa ayarı
st.set_page_config(page_title="Müşteri Destek Botu", page_icon="🤖")


# ✅ CACHE: Model sadece 1 kez yüklenir
@st.cache_resource
def get_embedding():
    return OpenAIEmbeddings(model="text-embedding-3-large")


@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")


@st.cache_resource
def load_vectordb(_embedding):
    return FAISS.load_local(DB_PATH, _embedding, allow_dangerous_deserialization=True)


def load_bm25_chunks():
    try:
        with open(BM25_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None, None


def build_prompt(query, docs):
    context = "\n\n".join([
        f"[Kaynak {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    return f"""Kurallar:
- Sadece verilen context'e göre cevap ver
- Context dışında bilgi kullanma
- Cevabı eksiltmeden ver, listeleri tam aktar
- Bilmediğin cevaplar için: "Bu konuda bilgim bulunmuyor, müşteri hizmetlerimize başvurabilirsiniz." de.

Context:
{context}

Soru: {query}

Cevap:"""


def hybrid_search(query, vectordb, bm25, chunks, k=5):
    vector_docs = vectordb.similarity_search(query, k=k)
    
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks[i] for i in bm25_top]
    
    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
    return list(combined.values())


def rerank(query, docs, reranker, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def process_pdf(uploaded_file, embedding):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()
    full_text = " ".join([p.page_content for p in pages])
    full_text = re.sub(r"\s+", " ", full_text)
    
    pattern = r'(\d{1,2}\.\s*Soru:.*?Cevap:.*?)(?=\d{1,2}\.\s*Soru:|$)'
    maddeler = re.findall(pattern, full_text)
    
    chunks = []
    for i, madde in enumerate(maddeler):
        soru_match = re.search(r"Soru:\s*(.*?)\s*Cevap:", madde)
        cevap_match = re.search(r"Cevap:\s*(.*)", madde)
        
        soru = soru_match.group(1).strip() if soru_match else ""
        cevap = cevap_match.group(1).strip() if cevap_match else ""
        
        chunks.append(Document(
            page_content=f"Soru: {soru}\nCevap: {cevap}",
            metadata={"madde_no": i + 1, "soru": soru, "cevap": cevap}
        ))
    
    # FAISS
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(DB_PATH)
    
    # BM25
    corpus = [doc.page_content for doc in chunks]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, chunks), f)
    
    return len(maddeler)


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cached modelleri al
embedding = get_embedding()
reranker = get_reranker()
llm = get_llm()

# Vector DB ve BM25 yükle
try:
    vectordb = load_vectordb(embedding)
    bm25, chunks = load_bm25_chunks()
    resources_loaded = bm25 is not None
except Exception as e:
    resources_loaded = False

# UI
st.title("🤖 Müşteri Destek Botu")
st.write("PDF yükleyin ve içeriğine dair sorular sorun.")

# PDF yükleme
uploaded_file = st.file_uploader("PDF dosyasını yükleyin", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        with st.spinner("PDF işleniyor..."):
            count = process_pdf(uploaded_file, embedding)
            
            # Cache'i temizle ve yeniden yükle
            load_vectordb.clear()
            vectordb = load_vectordb(embedding)
            bm25, chunks = load_bm25_chunks()
            resources_loaded = True
            
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.chat_history = []
        
        st.success(f"PDF yüklendi! {count} FAQ çıkarıldı.")

# Sohbet
if resources_loaded:
    user_question = st.text_input("Sorunuzu yazın:", key="question_input")
    
    if user_question:
        with st.spinner("Cevaplanıyor..."):
            docs = hybrid_search(user_question, vectordb, bm25, chunks, k=5)
            top_docs = rerank(user_question, docs, reranker, top_k=3)
            prompt = build_prompt(user_question, top_docs)
            response = llm.invoke(prompt)
            answer = response.content
            
            sources = "\n".join([
                f"- Kaynak {i+1}: Madde {doc.metadata.get('madde_no')}"
                for i, doc in enumerate(top_docs)
            ])
            
            full_response = f"{answer}\n\n📚 **Kaynaklar:**\n{sources}"
            
            st.session_state.chat_history.append(("Siz", user_question))
            st.session_state.chat_history.append(("Bot", full_response))
        
        # ❌ st.rerun() KALDIRILDI - Gerek yok!

# Sohbet geçmişi
if st.session_state.chat_history:
    st.subheader("💬 Sohbet Geçmişi")
    for sender, msg in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(msg)