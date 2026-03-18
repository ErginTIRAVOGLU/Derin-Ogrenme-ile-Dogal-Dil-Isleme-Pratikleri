"""
terminal üzerinden fastapi ile sürekli sohbet (post atarak)
api endpoint: /chat
"""
import requests # http istekleri yapmak için kullanılan kütüphane

API_URL = "http://127.0.0.1:8000/chat" # fastapi adresi

# başlangıçta kullanılan bilgiler
name= input("Adınız: ")
age= int(input("Yaşınız: "))

while True:
    user_msg = input(f"{name}: ")
    if user_msg.lower() in ["exit", "quit", "çıkış"]:
        print("Doktor Asistanı: Geçmiş olsun, iyi günler dilerim!")
        break

    payload = {
        "name": name,
        "age": age,
        "message": user_msg
    }
    
    try:
        res = requests.post(API_URL, json=payload, timeout=30)
        # res.raise_for_status() # hata varsa
        if res.status_code==200:
            print(f"Doktor Asistanı: {res.json()['response']}")
        else:
            print("hata", res.status_code, res.text)
    except requests.exceptions.RequestException as e:
        print(f"Hata: {e}")
    
