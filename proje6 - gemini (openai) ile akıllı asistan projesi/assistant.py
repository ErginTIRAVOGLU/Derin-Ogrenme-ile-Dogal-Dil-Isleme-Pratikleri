import os
from wsgiref import headers # ortam değişkenleri ve dosya yolu için
import requests # http istekleri için
from dotenv import load_dotenv # .env dosyasından ortam değişkenlerini yüklemek

# .env dosyasını yükle
load_dotenv();

api_key=os.getenv("OPEN_AI_KEY")

if not api_key:
    raise ValueError("Api Key bulunamadı!")

url="https://api.openai.com/v1/responses"

headers={
    "Content-Type":"application/json",
    "Authorization":f"Bearer {api_key}"
}

def get_response(prompt:str)->str: # prompt gönderip yanıt alan fonksiyon
    payload = {
        "model": "gpt-4.1-nano",
        "input":prompt
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            result=response.json()
            return result["output"][0]["content"][0]["text"]
        except Exception as e:
            return f"Yanıt hatası: {e}"
    else:
        return f"Hata: {response.status_code}: {response.text}"
    
# kullanıcı mesajına göre niyet sınıflandırması yapan fonksiyon
def detect_intent(message):
    prompt = f"""
                Kullanıcının aşağıdaki cümlesini sınıflandır:
                
                Etiketlerden sadece birini döndür:
                - not_ozet (eğer notları özetlemesini istiyorsa)
                - etkinlik_ozet (eğer etkinliklerin özetlenmesini istiyorsa)
                - normal (diğer her şey)
                
                Cümle: "{message}"
                Yalnızca etiket döndür: (örnek: not_ozet)
            """
    response=get_response(prompt)
    return response.strip().lower()



if __name__ == "__main__":
    user_input=input("Kullanıcı Sorusu: ") 
    response = get_response(user_input)
    print(f"ChatGPT: {response}")
