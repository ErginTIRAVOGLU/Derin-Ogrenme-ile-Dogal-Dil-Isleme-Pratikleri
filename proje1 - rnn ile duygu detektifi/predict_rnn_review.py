"""
Eğitilmiş rnn modelini (rnn_duygu_model.h5) kullanarak kullanıcı yorumlarını analiz edelim
"""


import numpy as np  
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# import tensorflow.keras.datasets as ds # dataset listesi
# #print(dir(ds))

max_features = 10000 # eğitim sırasında kullanılan maksimum kelime sayısı
maxlen = 500 # rnn modelinin beklediği sabit uzunluk => input_length

# stopwords (gereksiz kelimeler) listesi belirle ve sözcükleri hazırlama
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# imdb veri setinden kelimelerden index sözlüğü alındı (kelimelere kaşılık gelen index'leri aldık) 
word_index = imdb.get_word_index()

# sayı -> kelime sözlüğü oluşturma
index_to_word = {index +3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

# kelime -> sayı dönüşümü için sözlük
word_to_index = {word: index for index, word in index_to_word.items()}

# eğitim modelini yükle
model = load_model("rnn_duygu_model.h5")
print("Model başarıyla yüklendi.")

def predict_review(text):
    """
        kullanıcıdan gelen metni temizle, model uygun hale getir, tahmin sonucunu yazdır
    """
    
    # yorumu küçük harfli kelime listesine çevir
    words = text_to_word_sequence(text) # örn: This movie is great -> ["this", "movie", "is", "great"]
    
    # stopwords çıkarma ve sadece kelimeleri alma
    cleaned= [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]
    
    # her kelime eğitilen sözlükten sayıya çevrilir
    encoded = [word_to_index.get(word, 2) for word in cleaned] # 2(UNK) =
    
    # modelin beklediği sabit uzunluğa padding yapıyoruz
    padded = pad_sequences([encoded], maxlen=maxlen)
    
    # tahmin -> prediction (0 ile 1 arasında bir sonuç dönecek)
    prediction = model.predict(padded)[0][0]
    
    print(f"Pozitif tahmin olasılığı: {prediction:.4f}")
    if(prediction > 0.5):
        print("Pozitif duygu")
    else:
        print("Negatif duygu")
    
# konsole üzerinden kullanıcı girişi
user_review = input("Bir yorum giriniz: ")
predict_review(user_review)





