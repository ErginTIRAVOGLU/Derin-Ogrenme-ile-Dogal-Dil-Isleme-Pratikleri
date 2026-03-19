"""
Problem Tanımı: Gemini ile akıll asistan projesi -> notlar, görevler ve etkinlikler için akıllı asistan
    - Amaç : google gemini api (openai) tabanlı yapay zeka kullanan bir akıllı asistan geliştirme
    - Kullanıcının doğal dilde verdiği komutları anlar (sohbet botu)
    - Kural tabanlı olarak notlar ve etkinlikler oluşturalım
    - Akıllı asistanımız notlara ve etkinliklere erişim sağlayarak bize özetleme, bilgi çıkartma, takvim oluşturma gibi özellikler sunar
    

Model Tanıtımı: OpenAI (Eğitim Gemini ile yapıldı fakat ben OpenAI kullandım.) 
    - gpt-4.1-nano

API Tanımlama:

plan/program:
    - assistant: openai chatbot oluşturulur
    - database: sqlite veritabanı oluşturulur, notlar ve etkinlikleri depolamak lazım
    - main: bileşenleri bir araya getirir

install libraries

"""

from assistant import get_response, detect_intent
from database import initialize_db, get_notes, get_events, add_note, add_event

# veritabanı başlat
initialize_db()

# karşılama mesajı
print("Akıllı Asistana Hoşgeldiniz")
print("Komutlar: not ekle | etkinlik ekle | notları göster | etkinlikleri göster | sohbet et | çıkış")

# akıllı asistan
while True:
    komut=input("Komut girin: ").strip().lower()
    
    if komut == "not ekle":
        content=input("Not içeriği nedir? ")
        add_note(content)
        print("Not başarıyla kayıt edildli. ")
    elif komut =="etkinlik ekle":
        event = input("Etkinlik açıklaması:")
        date = input("Etkinlik tarihi? ")
        add_event(event, date)
        print("Etkinlik eklendi.")
    elif komut == "notları göster":
        notes = get_notes()
        if notes:
            print(f"Kaydedilmiş notlar")
            for content, created_at in notes:
                print(f"- [{created_at} {content}]")
        else:
            print("Kaydedilmiş not bulunamadı.")
    elif komut == "etkinlikleri göster":
        events = get_events()
        if events:
            print("Kaydedilmiş etkinlikler:")
            for event, event_date in events:
                print(f"- [{event_date}] {event}")
        else:
            print("Kaydedilmiş etkinlik bulunamadı.")
    elif komut=="çıkış":
        print("Çıkış yapıldı.")
        break
    elif komut=="sohbet et":
        message=input("Kullanıcı Sorusu: ").strip()
        intent=detect_intent(message) # kullanıcının niyetini anlama
        
        if(intent == "not_ozet"): # not özetleri
            notes = get_notes()
            if not notes:
                print("Kaydedilmiş not bulunamadı.")
                continue
            all_notes_text = "\n".join([f"- [{created_at}] {content}" for content, created_at in notes])
            prompt = f"Aşağıda bulunan notları özetler misin? \n\n {all_notes_text}"
            summary = get_response(prompt)
            print(f"ChatGPT: \n\n{summary}") 
        elif(intent == "etkinlik_ozet"): # etkinlik özetleri
            events = get_events()
            if not events:
                print("Kaydedilmiş etkinlik bulunamadı.")
                continue
            all_events_text = "\n".join([f"- [{event_date}] {event}" for event, event_date in events])
            prompt = f"Aşağıda bulunan etkinlikleri kullanıcı isteklerine göre özetler misin? \n\n {all_events_text} \n\n Kullanıcı isteği: {message}"
            summary = get_response(prompt)
            print(f"ChatGPT: \n\n{summary}")
        else: # normal sohbet
            reply=get_response(message)
            print(f"ChatGPT: {reply}")
    else:
        print("Geçersiz komut. Lütfen tekrar deneyin.")