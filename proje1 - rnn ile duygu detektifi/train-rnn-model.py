"""
RNN ile Duygu Detektifi (Sentiment Analysis)
Problem Tanımı: Bir yorumun olumlu mı olumsuz mu olduğunu anlamak. (classification problem)
    IMDB film yorumları veri seti ile bir metnin duygusal analizini gerçekleştirme.
    - this movie is awesome -> pozitif
    - it was terrible movie -> negatif


RNN: Tekrarlayan sinir ağları: siralı veriler üzerinde çalışıyor, metin gibi verilerle önceki bilgileri hatırlayarak sonraki tahminleri yapmaya çalışırlar
Girdi:   film -> çok ->  kötüydü
Bellek:
Çıktı:  anlam   anlam   olumsuz

Veri Seti: IMDB veri seti: film yorumları ( olumlu ve olumsuz)
    - 50000 adet film yorumu
    - 0 negatif, 1 pozitif
    - great = 65 (token şeklinde gelecek)


plan/program:

Gerekli Kurulumlar:
    pip install tensorflow matplotlib nltk
 
import Libraries:
"""

# import libraries
import numpy as np
import nltk # natural language tool kit
import matplotlib.pyplot as plt
from nltk.corpus import stopwords # gereksiz kelime listesi
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences

# source venv/bin/activate
# . venv/bin/activate
# pip install tensorflow matplotlib nltk

# stopwords (gereksiz kelimeler) listesi belirle
nltk.download("stopwords") # nltk içinden ingilizce stopwords indiriliyor
stop_words = set(stopwords.words("english")) # küçük ve anlamsız kelimeler ayıklanacak

# model parametreleri
max_features = 10000 # en çok kullanılan 10 bin kelime
movie_index=0
maxlen=500

# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features) # train/test ayrılmış şekilde verilir


# örnek veri incelemesi
original_word_index = imdb.get_word_index()

# sayı kelime dönüşüm sözlüğü hazırlama
inv_word_index = {index +3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>" # 0: boşluk / padding
inv_word_index[1] = "<START>" # 1: cümle başlangıcı
inv_word_index[2] = "<UNK>" # 2: bilinmeyen kelime
# inv_word_index[3] -> great : 65

# sayı dizisini kelimelere çeviren fonksiyon
def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])


# ilk eğitim verisini yazdıralım
print("ilk yorum: (sayı dizisi)")
print(X_train[movie_index])

print("ilk yorum: (kelime dizisi)")
print(decode_review(X_train[movie_index]))

print(f"Label: {"Pozitif" if y_train[movie_index]==1 else "Negatif"}")

# gerekli sözlüklerin oluşturulması: word to index ve index to word
word_index = imdb.get_word_index()
index_to_word = {index +3: word for word, index in word_index.items()} # sayılardan kelimeler
index_to_word[0] = "<PAD>" # 0: boşluk / padding
index_to_word[1] = "<START>" # 1: cümle başlangıcı
index_to_word[2] = "<UNK>" # 2: bilinmeyen kelime

word_to_index = {word: index for index, word in index_to_word.items()} # kelimerden sayılarak


# data preprocessing (veri ön işleme)
def preprocess_review(encoded_review):
    # sayıları kelimelere çevir
    words = [index_to_word.get(i, "?") for i in encoded_review if i>=3]
    
    # sadece harflerden oluşan ve stop words olmalayanları al
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]
    
    # tekrardan temizlenmiş metni sayılara çevir
    
    return [word_to_index.get(word, 2) for word in cleaned]

# veriyi temizle ve sabit uzunluğu pad et
X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

# pad sequence
"""
merhaba bugün hava çok güzel
merhaba, naber, 0, 0, 0
"""

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

    
# RNN Modeli oluşturma
model = Sequential(); # base model: katmanları sıralı olarak eklemek için

# embedding katmanı: kelime indexlerini 32 boyutlu bir vektöre dönüştürür
model.add(Embedding(input_dim = max_features, output_dim = 32, input_length = maxlen))

# simpleRnn katmanı: metni sırayla işler ve bağlam ilişkisini öğrenir
model.add(SimpleRNN(units = 32)) # cell (nöron) sayısı

#output katmanı: binary classification (sınıflandırma): sigmoid, 1 nöron
"""
negatif -> 0.7 // 1 nöron kullanıldı.
"""
model.add(Dense(1, activation = "sigmoid"))

# model compile
model.compile(
    optimizer = "adam", # ağırlık güncellemesi için kullanılan algoritma
    loss = "binary_crossentropy", # kayıp fonksiyonu
    metrics = ["accuracy"] # değerlendirme metriği
)

print(model.summary())

# modelin eğitimi
# training (modelin eğitimi)
history = model.fit(
    X_train, y_train, # girdi ve çıktı verisi
    epochs = 2, # eğitim tekrar sayısı ( Tüm veriyi 2 kere eğit )
    batch_size = 64, # torba ( aynı anda işlenecek örnek sayısı yani 64 'lü paketler/torbalar halinde işle (50000 tane verimiz var 50000 tane veriyi aynı anda işleyemeyiz))
    validation_split = 0.2 # %20 doğrulama için ayır
)


# model evaluation
def plot_history(hist):
    plt.figure(figsize=(12,4))
    
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(hist.history["accuracy"], label="Train")
    plt.plot(hist.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # loss plot
    plt.subplot(1,2,2)
    plt.plot(hist.history["loss"], label="Train")
    plt.plot(hist.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# test verisiyle modeli değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test) # test
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}") # test

# eğitilen modelin kaydını yapalım
model.save("rnn_duygu_model.h5")
print(f"Model kaydedildi: rnn_duygu_model.h5")