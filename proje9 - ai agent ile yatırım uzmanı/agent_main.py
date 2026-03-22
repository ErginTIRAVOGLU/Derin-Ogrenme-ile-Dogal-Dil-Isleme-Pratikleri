"""
AI Agent ile Yatırım Danışmanı

problem tanımı: doğal dilde yazılmış yatırım sorularını anlayıp, 
                doğru kaynaklardan bilgi toplayarak, yatırımcıya net özet ve güncel yanıtlar sunabilen, 
                akıllı bir yatırım danışmanı
                - dolar bugün kaç tl
                - apple hissesi ne kadar
                - altın hakkında son haberler nedir?
                - tesla hissesi mi apple hissesi mi almak daha mantıklı?

hedefler:
    - kullanıcıdan gelen doğal dil sorularını analiz etmek,
    - gerekli olması durumunda internetten arama yapmak, kur çevirisi gerçekleştirmek, hisse bilgisi almak
    - bilgileri birleştirmek, özetlemek ve sadece bir dilde yazmak
    - bunları profesyonel bir dil ile yapmak
    

teknolojiler:
    - langchain: agent, tool, llm kontrolü ve iş akışı sağlamak
    - openai: llm modeli (gpt-4.1-nano)
    - coingecko api: usd -> try kur çevrimi
    - finnhub: hisse senedi bilgileri (api key ile)
    - duckduckgo: arama motoru

plan/program:
    - araçlar (tools): duckduckgo (arama motoru ile arama), coingecko api (kur çevirimi), Finnhub api (hisse senedi bilgileri)
    - llm modeli oluştur
    - agent oluştur
    - agent ile kullanıcıdan gelen soruları cevapla

install libraries: freeze

Sonrasonda:
    - memory ekle
    - streamlit ile kullanıcı arayüzü oluştur
    - fastapi ile api oluştur
    - plan and execute
    - rag
"""

 
from doctest import debug
from tabnanny import verbose

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from tools.search_tool import search
from tools.currency_converter import convert_usd_to_try
from tools.market_api import get_stock_info

from dotenv import load_dotenv
import os


from langchain_core.prompts import ChatPromptTemplate # kişiselleştirilmiş yatırım uzmanı için prompt template
 


# .env dosyasını oku
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# llm
llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.7,
        api_key=api_key
    )

@tool
def search_tool(query: str) -> str:
    """Güncel haberleri ve bilgileri internetten arar"""
    return search.invoke({"query": query})

@tool
def currency_tool(amount: float) -> str:
    """USD -> TRY çevirimi yapar"""
    return str(convert_usd_to_try.invoke({"amount": amount}))

@tool
def stock_tool(symbol: str) -> str:
    """Hisse senedi bilgisi verir (örn: AAPL, TSLA)"""
    return get_stock_info.invoke({"ticker": symbol})


# araçlar listesi (agents)
tools= [
    search_tool,
    currency_tool,
    stock_tool
]

system_prompt= """
        Sen deneyimli ve güvenilir bir yatırım danışmanısın.
        Amacın, kullanıcının finansal ve yatırım konularındaki sorularını anlamak,
        doğru araçları (tools/agents) kullanarak analiz etmek ve 
        sonuçları net, sakin ve profesyonel bir dille sunmak.
        
        Araçlar:
        - Döviz Çevirici (USD->TRY)
        - Hisse Bilgisi Sorgulayıcı (örn: AAPL, TSLA)
        - Arama Motoru DuckDuckGo (güncel haberler, analiz ...)
        
        Kurallar:
        1- Soruyu analiz etmeden hemen cevap verme.
        2- Gerekirse birden fazla tool kullan
        3- Kullanıcıya yatırım kararı verdirme sadece bilgi ver.
        4- Yanıtlarında kısa açıklamalar, sayılar veriler ve açıklayıcı cümleler kullan.
        
     """

agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
    debug=False
)

if __name__ == "__main__":

    while True:
        user_input = input("Sorunuz >> ")

        if user_input.lower() in ["exit", "quit"]:
            break
        
        
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })

        print("\n🤖:", response["messages"][-1].content)
        