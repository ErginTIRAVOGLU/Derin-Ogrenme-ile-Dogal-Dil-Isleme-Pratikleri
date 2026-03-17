"""
Problem tanımı: yorumlardan yıldız puan tahmini (1-5), regresyon problemi çözeceğiz.
    - çok iyiydi çok memnun kaldım -> 4.5
    - berbattı, bir daha gelmem -> 1.2
    

Veri seti: yelp dataset, hugging face'den indireceğiz (restoran, doktor, otel, araba yıkama,... yorumları)
    - text: yorum metni
    - label: 0-4 arasında değişiyor ama biz bunu 1 ile 5 arasına çekeceğiz
    - https://huggingface.co/datasets/Yelp/yelp_review_full

LSTM: long short term memory -> klasik, paralel, tekrarlayan sinir ağı yapısının geliştirilmiş hali (özellikle uzun metinlerde, kelimeler arası ilişkileri anlamada ve duygusal tonlamaları yakalamada çok iyi)
    - Bir yorumu baştan sona okur, sonrasında yorumun genel anlamına karşılık gelen yıldız puanını çıkarır

install libraries: freeze requirements.txt

plan/program

import libraries:
"""


# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizer'i diske kaydetmek için kullanacağız

from sklearn.model_selection import train_test_split # veriyi eğitim ve test olmak üzere 2 ye ayır
from sklearn.preprocessing import MinMaxScaler # normalization

from tensorflow.keras.preprocessing.text import Tokenizer # tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError 
from tensorflow.keras.metrics import MeanAbsoluteError # 5 çıkması gereken değeri 4 buldunuz, fark =1, 4 çıkması gereken değeri 5 buldunuz, fark =1 -> 1+1=2 2/2=1




# load yelp dataset
# hugging face den yelp veri setini yükleme
splits = {"train":"yelp_review_full/train-00000-of-00001.parquet"}
train_path = "hf://datasets/Yelp/yelp_review_full/" + splits["train"]

#parquet formatından veriyi pandas ile oku
df = pd.read_parquet(train_path)
print(df.head())

# etiketleri 0-4 aralığından 1-5 aralığına dönüştür
df["label"]=df["label"]+1

# data preprocessing
texts = df["text"].values # yorum metinleri
labels = df["label"].values # puanlar 1-5 arasında

# tokenizer: metni sayıya çevir
# num_words: en çok geçen ilk 10000 kelime
# OOV: bilinmeyen kelimleri bu etiketle göster
tokenizer=Tokenizer(num_words=10000, oov_token="<UNK>")

# metni sayılara dönüştür
tokenizer.fit_on_texts(texts)

# tokenizeri diske kaydet
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)


# yorumları dizi haline getir
sequences = tokenizer.texts_to_sequences(texts)

# tüm dizileri sabit uzunluğa getir yani padding uygula (kısa olan cümledeki boş kalan kelimlerin yerine 0 ile doldur)
padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# etiketler 1 ile 5 arasında, normalization ile 0 ile 1 arasına alalım, çünkü regresyon problemlerinde daha stabil bir öğrenme sağlıyor
scaler=MinMaxScaler() # 1 ile 5 'ten -1 çıkart = 0-4 sonra /4 = 0-1
labels_scaled = scaler.fit_transform(labels.reshape(-1,1)) 

# eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train: {X_train[:2]}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train: {y_train[:2]}")




# LSTM tabanlı regresyen modeli

# model compile and trainig

# eğitim kayıp grafiğini görselleştir ve modeli kaydet