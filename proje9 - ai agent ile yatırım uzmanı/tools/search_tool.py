"""
duckduckgosearchrun, langchain kütüphanesi içerisinde gelen hazır bir araç
web araması için duckduckgo motoru kullanılacak
"""

from langchain_community.tools  import DuckDuckGoSearchRun



# duckduckgo arama aracının bir örneği
search = DuckDuckGoSearchRun()


if __name__ == "__main__": # test amaçlı
    # aranacak terimi belirle
    query = "what is the santigrat degree temperature today in istanbul"
    
    # arama motoruna sorgu gönder ve sonucu al
    result = search.invoke(query)
    
    # sonucu yazdır
    print(f"Arama sonucu: \n {result}")
    
