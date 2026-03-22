"""
finhub api ile hisse senedi bilgilerini al
api_key 
"""

from langchain.tools import tool # bu dekoratör sayesinde fonksiyonumuz bir langchain aracı olarak tanımlanabilecek 
import requests # http istekleri
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def get_stock_info(ticker:str) ->str: # güncel hisse senedi bilgileri
    """Hisse senedi fiyatını getirir.
    
    Args:
        symbol: Hisse senedi sembolü (örneğin: AAPL, MSFT, TSLA)
    
    Returns:
        Güncel hisse senedi fiyatı ve temel bilgiler
    """
    try:
        api_key = os.getenv("FINHUB_API_KEY")
        if not api_key:
            return "API anahtarı bulunamadı"
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            return f"Api Hatası: {response.status_code}"
        
        
        data = response.json()
        
        """
            c = güncel fiyat
            o = açılış fiyatı
            h = en yüksek
            l = en düşük
        """
        
        current = data.get("c")
        opening = data.get("o")
        high = data.get("h")
        low = data.get("l")

        return(
            f"{ticker} Hisse Bilgisi: \n"
            f"Güncel Fiyat: {current} USD\n"
            f"Açılış Fiyatı: {opening} USD\n"
            f"En Yüksek: {high} USD\n"
            f"En Düşük: {low} USD\n"
        )
    except Exception as e:
        return f"Hata: {e}"
  

if __name__ == "__main__":
    print(get_stock_info.run({"ticker":"GOOGL"}))

        

