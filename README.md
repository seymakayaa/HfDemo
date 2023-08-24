# HfDemo

```python
# Kütüphaneden sınıflar içeri aktarılır
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
```


```python
# BERT tabanlı, duygu analizi için özelleştirilmiş model 
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

```python
# Tokenizer: Metin verilerini modele girdi olarak verebilmek için tokenlere çevirmeyi sağlar
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model, metin sınıflandırma görevleri için kullanılmak üzere hazır hale geldi
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

```python
# Tahmin yapılacak metin
text = "This film is fantastic!"
```

```python
# Metni tokenizer ile tokenlere ayrılır, return_tensors="pt" ile sonuç PyTorch tensor formatında alınır, padding=True ve truncation=True ile metin tokenlerinin boyutları yarlanır.
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
```


```python
# İnceleme yaparken gradyanları hesaplamamak için, Tahmin yaparken gradyan hesaplaması yapılmaz çünkü tahmin işlemi, modelin eğitimi ile ilgili olmayan ve sadece mevcut parametrelerle girdiye dayalı sonuçlar üretme amacı taşır. Gradyanlar, modelin parametrelerini eğitim sırasında güncellemek için kullanılır, yani modelin daha iyi öğrenmesini sağlamak amacıyla kullanılır.
with torch.no_grad():
    outputs = model(**inputs)
```


```python
#  Logitler, farklı sınıfların tahminlerini içeren sayısal değerlerdir.
logits = outputs.logits
print(logits)
```


```python
# Logitleri olasılıklara dönüştürmek için
probabilities = torch.softmax(logits, dim=-1)
print(probabilities)
```


```python
# En yüksek olasılığa sahip sınıfı belirleme, olasılık değerlerinin en yüksek olduğu sınıfın indeksini belirlemek için kullanılır.verilen tensörde en büyük değeri ve ilgili indeksi döndürür.
predicted_class = torch.argmax(probabilities, dim=-1)
print(predicted_class)
```


```python
# Duygu tahminlerini ekrana yazdırma
emotion_labels = ["Çok Olumsuz", "Olumsuz", "Nötr", "Olumlu", "Çok Olumlu"]
predicted_emotion = emotion_labels[predicted_class.item()]
print("Metnin Tahmin Edilen Duygusu:", predicted_emotion)
```
