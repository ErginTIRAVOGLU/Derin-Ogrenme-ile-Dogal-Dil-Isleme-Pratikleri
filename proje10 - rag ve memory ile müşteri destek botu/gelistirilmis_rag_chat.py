import pickle
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

load_dotenv()

DB_PATH = "rag_vectorstore_gelistirilmis"
BM25_PATH = "bm25.pkl"

# LOAD VECTOR DB
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

# LOAD BM25
with open(BM25_PATH, "rb") as f:
    bm25, chunks = pickle.load(f)

# RERANKER
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# LLM
llm = ChatOpenAI(model="gpt-4.1-nano")

# MEMORY
chat_history = []

# HYBRID SEARCH
def hybrid_search(query, k=5):
    vector_docs = vectordb.similarity_search(query, k=k)

    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks[i] for i in bm25_top]

    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}

    return list(combined.values())

# RERANK
def rerank(query, docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored[:top_k]]

def get_response_text(response):
    if isinstance(response.content, str):
        return response.content
    
    if isinstance(response.content, list):
        texts = []
        for item in response.content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return " ".join(texts)
    
    return str(response.content)

# PROMPT
def build_prompt(query, docs, history):
    context = "\n\n".join([
        f"[Kaynak {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    history_text = "\n".join(history[-5:])

    return f"""
    Kurallar:
    - Sadece verilen context'e göre cevap ver
    - Context dışında bilgi kullanma
    - Önceki konuşmaları ASLA kullanma
    - Cevabı eksiltmeden ver, listeleri tam aktar
    - Bilmediğin cevaplar için yada eğer context'te cevap yoksa yada bağlamda hiç ilgili bilgi yoksa: 
        "Bu konuda bilgim bulunmuyor, müşteri hizmetlerimize başvurabilirsiniz." de.

    Context:
    {context}

    Soru: {query}

    Cevap:
    """

# CHAT
def ask(question):
    global chat_history

    docs = hybrid_search(question, k=5)

    print("\n🔍 HYBRID SONUÇ:")
    for i, d in enumerate(docs):
        print(f"[{i+1}] {d.page_content[:80]}...")

    top_docs = rerank(question, docs, top_k=3)

    print("\n🏆 RERANK SONUÇ:")
    for i, d in enumerate(top_docs):
        print(f"[{i+1}] {d.page_content[:80]}...")

    prompt = build_prompt(question, top_docs, chat_history)

    response = llm.invoke(prompt)

    chat_history.append(f"Kullanıcı: {question}")
    chat_history.append(f"Bot: {response.content}")

    sources = "\n".join([
        f"- Kaynak {i+1}: madde {doc.metadata.get('madde_no')}"
        for i, doc in enumerate(top_docs)
    ])

    answer_text = get_response_text(response)

    return answer_text + "\n\n📚 Kaynaklar:\n" + sources


# LOOP
print("🤖 Müşteri destek botuna hoş geldiniz!")
while True:
    q = input("Soru: ")
    if q.lower() in ["exit", "quit"]:
        break

    answer = ask(q)
    print("\nCevap:", answer)