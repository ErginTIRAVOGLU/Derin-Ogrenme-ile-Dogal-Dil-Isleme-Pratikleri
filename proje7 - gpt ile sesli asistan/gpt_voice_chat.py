"""
Problem tanımı: gpt ile sesli sohbet
    - kullanıcının mikrofona konuşarak soru sorması
    - openai whisper modeli
    - metnin gpt-4.1-nano ile analiz edilmesi
    - güvenlik mekanizması: zararlı dil filtreleme

Kullanılan teknolojiler:
    - ses kaydı
    - ses -> metin: openai whisper modeli
    - cevap üretimi: openai gpt-4.1-nano
    - loglama: logging modülü
    - zararlı içerik filtreleme: re kütüphanesi
    

model tanıtımı: openai whisper ve gpt-4.1-nano
            - çok dilli konuşma modeli
            - konuşmaları yazıya döker
            - birden çok dili destekler
            - otomatik dil algılama yapabilir
            - whisper 1 modeli
            - whisper light:
                * hafif, hızlı, offline ve açık kaynaklı

api tanımlama: 

plan/program: 

install libraries: freeze

import libraries:

"""

# import libraries

from genericpath import exists
from json import load
from math import e
from random import sample

from openai import OpenAI # OpenAI Api client
import sounddevice as sd # mikrofon erişimi için sound device kütüphanesi
from scipy.io.wavfile import write # wav formatında ses kaydı için scipy kütüphanesi
import os 
import uuid # unique id
import re # zararlı içerik filtreleme
from datetime import datetime 
from dotenv import load_dotenv
import logging # loglama


# log ayarlari
now=datetime.now().strftime("%Y_%m_%d") # dosyaadı için şuanki zaman
logfile=f"logs/konusma_{now}.log" # log dosyası

os.makedirs("logs",exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(logfile, encoding="utf-8"),
        logging.StreamHandler() # konsola yazdirma
    ]
)

logger=logging.getLogger(__name__)

# .env oluştur ve yükle

load_dotenv() # .env dosyasını yükle

client = OpenAI()

DURATION =5 # tek seferde kaç saniye kayıt alacağımızın parametresi
FS=44100 # örnekleme frekansı

 

# zararlı sözcük filtreleme

BANNED_WORDS=["zararlı"] # zararlı sözcükler

def filter_bad_words(text):
    filtered_text=text
    for word in BANNED_WORDS:
        if re.search(word,text,re.IGNORECASE):
            logger.warning("Zararlı kelime tespit edildi: %s",word)
        filtered_text=re.sub(word,"***",filtered_text,flags=re.IGNORECASE)
    return filtered_text

# filter_bad_words("merhaba, zararlı bir kelime var mı?") # test


# ses kaydı alma 

def record_audio(filename="recorded.wav",duration=DURATION): # mikrofon kaydı al
    logger.info("Ses kaydı başlatılıyor...")
    recording=sd.rec(int(duration*FS),samplerate=FS, channels=1)
    sd.wait()    
    write(filename,FS,recording)
    logger.info("Ses kaydı tamamlandı.")

# whisper ayarları, sesi metne çevirme
def transcribe_with_whisper(audio_path): # sesi metne çevirme
    logger.info("Ses metne dönüştürülüyor...")
    with open(audio_path,"rb") as audio_file: # ses dosyasini ac
        transcript=client.audio.transcriptions.create(
            model="whisper-1", # model
            file=audio_file,
            language="tr" # dil       
        )
    return transcript.text # metni döndür

# llm/dil modeli oluşturma
def get_gpt_response(messages): # gonderilen mesaja göre yanıt veren llm fonksiyonu
    logger.info("GPT yanıt bekleniyor...")
    response=client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
    )
    return response.choices[0].message.content # ilk olasilik cevanı döner

print(f"Gpt Yanıt : {get_gpt_response([{'role':'user','content':'Merhaba'}])}")


# hepsini birleştir ve çalıştır
if __name__=="__main__":
    logger.info("--- --- GPT Sesli Chatbot Başlatılıyor --- ---")
    logger.info(f"Konuşma log dosyası: {logfile}")
    
    # mesaj geçmişini sistem mesajıyla baslat
    messages= [
        {"role":"system","content":"Sen yardımsever bir sesli asistansın, konuşmalara uygun cevap ver."}
    ]
    
    while True: # kullanıcı çıkana kadar sonsuz bir döngü oluştur.
        uid=str(uuid.uuid4())
        audio_file=f"record_{uid}.wav" # geçici wav dosyasi ismi
        
        record_audio(audio_file,DURATION) # mikrofon kaydı al
        question = transcribe_with_whisper(audio_file) # sesi metne cevir
        logger.info(f"Soru: {question}")
        
        filtered_question=filter_bad_words(question) # zararlı kelimeleri filtrele
        if filtered_question != question:
            logger.info(f"Filtrelenmiş soru: {filtered_question}")
        
        if "çık" in filtered_question.lower():
            logger.info("Çıkış komutu algılandı, program kapatılıyor.")
            break
        
        messages.append({"role":"user", "content":filtered_question}) # kullanıcı mesajını ekle
        answer = get_gpt_response(messages) # gpt yanıtını al
        
        logger.info(f"Cevap: {answer}")
        
        os.remove(audio_file) # geçici dosyayı sil
    
    logger.info("--- --- GPT Sesli Chatbot Sonlandırıldı --- ---")
    
    