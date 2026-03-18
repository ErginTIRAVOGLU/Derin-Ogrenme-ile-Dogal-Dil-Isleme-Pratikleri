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
model = Sequential()

# embedding katmanı: kelime indexlerini vektör uzayına dönüştürür
# input_dim: 10000 -> kelime sayısı
# output_dim: 128 -> her bir kelime 128 boyutlu bir vektörle temsil edilecek
# input_length: 100 -> sabit dizi uzunluğu yani her bir metnimizin uzunluğu
model.add(Embedding(input_dim=10000, output_dim=128,input_length=100))

# LSTM katmanı: sıralı veride bağlamı öğrenecek olan katman
model.add(LSTM(128)) # 128: lstm de bulunan hücre sayısı yani daha fazla öğrenme kapasitesi
# 520000 kayıt var 128 yeterli olacaktır, olmaz ise arttırılacak

# tam bağlı (dense) katmanı:
model.add(Dense(64, activation="relu"))

# output katmanı:
# relu ve tanh -> tam bağlı katmanlarda kullanılıyor, yani ara/hidden layerlarında kullanılıyor
# sigmoid ve softmax -> sınıflandırma problemlerinde kullanılıyor, sigmoid 2 sınıflı sınıflandırma problemleri için, softmax çok sınıflı sınıflandırma problemlerinde kullanılıyor
# linear -> regresyon problemlerinde kullanılıyor 
model.add(Dense(1,activation="linear")) # relu, tanh, sigmoid, softmax ve linear


# model compile and trainig
model.compile(
    optimizer="adam", # Adam: öğrenme hızını otomatik ayarlayan, hızlı ve stabil bir optimizasyon algoritmasıdır.
    loss=MeanSquaredError(), # MSE: hataların karesini alır, büyük hataları daha fazla cezalandırdığı için regresyonda yaygın kullanılır.
    metrics=[MeanAbsoluteError()] # MAE: tahminlerin ortalama kaç birim saptığını gösterir, yorumlaması kolay bir performans metriğidir.
)

history = model.fit(
    X_train, y_train,
    epochs=3, # model veriyi 3 kez baştan sona öğrenir (her tur = 1 epoch)
    batch_size=64, # veriyi 64'lük parçalara bölerek öğrenir (daha hızlı ve stabil eğitim)
    validation_split=0.2 # verinin %20'sini doğrulama için ayırır, modelin genelleme performansını ölçer
)

# eğitim kayıp grafiğini görselleştir ve modeli kaydet
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation loss - Eğitim süreci: MSE:")
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.show()

# 6500/6500 loss: 0.0346 - mean_absolute_error: 0.1400 - val_loss: 0.0360 - val_mean_absolute_error:0.1441 
# 6500/6500 # toplam 6500 batch işlenmiş (örnek: 416.000 veri / 64 batch_size ≈ 6500 batch)

# loss: 0.0346 # eğitim verisindeki MSE hatası (örnek: gerçek=3, tahmin=2 → hata=1 → karesi=1, ortalamada 0.0346)
# mean_absolute_error: 0.1400 # eğitim verisinde tahminler ortalama 0.14 birim sapıyor (örnek: 3 yerine 2.86 veya 3.14 tahmin)
# val_loss: 0.0360 # doğrulama verisindeki MSE (örnek: modelin hiç görmediği veride hata biraz artmış ama yakın)
# val_mean_absolute_error: 0.1441 # doğrulama verisinde ortalama sapma 0.1441 (örnek: 5 yerine 4.85 veya 5.15 tahmin)

# modeli kaydet
model.save("regression_lstm_yelp.h5")