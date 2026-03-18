"""
yorum yazalım, lstm bu yoruma göre bir tahmin ortay çıkarsın
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# modeli yükle
model = load_model("regression_lstm_yelp.h5", compile=False)

# tokenizeri yükle
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)



# örnek input "ben bir doktora gittim ve bu doktoru çok sevdim."
# 1 star - 5 star
texts = [
    "Terrible. Preordered my tires and when I arrived they couldn't find the order anywhere. Once we got through that process I waited over 2 hours for them to be put on... I was originally told it would take 30 mins. Slow, over priced, I'll go elsewhere next time.",
    "BEST DINER IN THE COUNRTY!!! We've been to many famous diners across the country and we still give Gab and Eat the best rating !! I was a little intimidated when I first walked in and there was like 2 pounds of butter just sitting on top of the homefires on the grill. If you are looking for a healthy breakfast they probably can accomodate you, but Everything I ate was clearly the opposite of healthy. After trying like every meal they have I would recommend the mix grill(half unless you are sharing) adding cajun seasoning with the texas toast. Burgers are great too.\n\nIt's hard to find a place that makes a better breakfast than you could make by yourself at home, but this place does it. The atmosphere is classy, old school Americana.",
    "Best experience ever",
    "worst experience ever"
]

# PREPROCESSING
# Tokenizer "0 1 2 3 4 5 6 7 8"
# padding "0 1 2 3 4 5 6 7 8 0 0 0 0 0 0 0 0 0 0 ... 0" # boyu 100 olacak / embedding katmanı ->  input_length
# text'i sayılara çevir ve padding işlemini gerçekleştir
sequences = tokenizer.texts_to_sequences(texts)
padded= pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# PREDICTION
# "LSTM prediction"
predictions = model.predict(padded)
print(predictions)


# POST PROCESSING
predictions_scaled = predictions * 5

# sonuçları yazdır
for i, comment in enumerate(texts):
    print(f"Yotum \n{comment}")
    print(f"Tahmini skor değeri: {predictions_scaled[i][0]:.2f}")
