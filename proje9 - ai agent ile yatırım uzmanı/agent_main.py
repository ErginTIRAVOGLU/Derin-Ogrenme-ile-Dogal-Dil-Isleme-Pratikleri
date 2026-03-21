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



