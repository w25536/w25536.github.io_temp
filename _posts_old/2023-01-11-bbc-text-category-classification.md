---
layout: page
title: "bbc-text.csv 데이터셋을 활용한 BBC 뉴스 아티클 카테고리 분류기 만들기"
description: "`bbc-text.csv` 데이터셋을 활용한 BBC 뉴스 아티클 카테고리 분류기 만들기에 대해 알아보겠습니다."
headline: "`bbc-text.csv` 데이터셋을 활용한 BBC 뉴스 아티클 카테고리 분류기 만들기에 대해 알아보겠습니다."
categories: tensorflow
tags: [python, 파이썬, bbc-text.csv, tensorflow certificate, 텐서플로 자격증, 자연어 처리, lstm, embedding, stopwords, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-11
---

BBC 뉴스 아티클 묶음 데이터셋인 `bbc-text.csv` 파일을 활용하여 TensorFlow 의 Tokenizer로 단어 사전을 만들고 자연어 처리 모델 학습을 위한 데이터 전처리를 진행해 보겠습니다. `bbc-text.csv` 파일을 pandas로 읽어와서 데이터프레임 변환 후 라벨 인코딩을 포함한 간단한 전처리를 다룹니다.

문장 데이터(text) 전처리에서는 **토크나이저 생성, 단어 사전 생성, 불용어(stopwords) 처리, 시퀀스 변환** 등을 다룹니다. 

모델링에서는 `Embedding layer`와 `Bidirectional LSTM`으로 BBC 뉴스 아티클의 뉴스 카테고리 분류기를 생성하겠습니다.

<u>(본 예제는 텐서플로 자격 인증 시험(TensorFlow Developers Certificate)의 기출 문제 중 하나를 다뤄 본 튜토리얼입니다.)</u>



**Dataset Reference**

```
About this file

Source data from public data set on BBC news articles:
D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. [PDF] [BibTeX].

http://mlg.ucd.ie/datasets/bbc.html

Cleaned up version exported to https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
```




<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 필요한 모듈 import



```python
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
```

데이터셋 다운로드



```python
url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
urllib.request.urlretrieve(url, 'bbc-text.csv')
```



다운로드 받은 `bbc-text.csv` 파일을 읽어서 df 변수에 로드 합니다.



```python
df = pd.read_csv('bbc-text.csv')
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tech</td>
      <td>tv future in the hands of viewers with home th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>business</td>
      <td>worldcom boss  left books alone  former worldc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sport</td>
      <td>tigers wary of farrell  gamble  leicester say ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sport</td>
      <td>yeading face newcastle in fa cup premiership s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>entertainment</td>
      <td>ocean s twelve raids box office ocean s twelve...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2220</th>
      <td>business</td>
      <td>cars pull down us retail figures us retail sal...</td>
    </tr>
    <tr>
      <th>2221</th>
      <td>politics</td>
      <td>kilroy unveils immigration policy ex-chatshow ...</td>
    </tr>
    <tr>
      <th>2222</th>
      <td>entertainment</td>
      <td>rem announce new glasgow concert us band rem h...</td>
    </tr>
    <tr>
      <th>2223</th>
      <td>politics</td>
      <td>how political squabbles snowball it s become c...</td>
    </tr>
    <tr>
      <th>2224</th>
      <td>sport</td>
      <td>souness delight at euro progress boss graeme s...</td>
    </tr>
  </tbody>
</table>
<p>2225 rows × 2 columns</p>
</div>


## Label 값 확인


`category` 종류 확인



```python
df['category'].value_counts()
```

<pre>
sport            511
business         510
politics         417
tech             401
entertainment    386
Name: category, dtype: int64
</pre>
위의 `value_counts()` 함수로 label의 value는 `sport, `business`, `politics`, `tech`, `entertainment` 이렇게 5가지의 종류가 존재합니다.



하지만, TensorFlow Certificate 시험에서는 아래와 같은 가이드라인을 줍니다.


```

PLEASE NOTE -- WHILE THERE ARE 5 CATEGORIES, THEY ARE NUMBERED 1 THROUGH 5 IN THE DATASET

SO IF YOU ONE-HOT ENCODE THEM, THEY WILL END UP WITH 6 VALUES, SO THE OUTPUT LAYER HERE

SHOULD ALWAYS HAVE 6 NEURONS AS BELOW. MAKE SURE WHEN YOU ENCODE YOUR LABELS THAT YOU USE

THE SAME FORMAT, OR THE TESTS WILL FAIL

0 = UNUSED

1 = SPORT

2 = BUSINESS

3 = POLITICS

4 = TECH

5 = ENTERTAINMENT

```


즉, 5개의 카테고리가 존재하는 것은 맞지만 0번 label 에는 `UNUSED` 항목을 남겨 놓고, **1~5번 라벨에 맵핑되는 카테고리를 규정**하고 있습니다.



반드시 위의 번호에 맞게 라벨 인코딩을 해줘야 채점서버에서 올바르게 채점을 진행할 수 있기에, **라벨 인코딩시 위의 규정에 따라 인코딩을 진행**하여야 합니다.



```python
# category encoding map
m = {
    'unused': 0, 
    'sport': 1, 
    'business': 2, 
    'politics': 3, 
    'tech': 4, 
    'entertainment': 5
}

# map 함수로 인코딩 변환
df['category'] = df['category'].map(m)
df['category'].value_counts()
```

<pre>
1    511
2    510
3    417
4    401
5    386
Name: category, dtype: int64
</pre>
0번 라벨 값은 없었기 때문에 0번을 제외한 **1~5번까지 올바르게 출력됨을 확인**할 수 있습니다.



```python
# hyperparameter settings
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 2000

# 불용어 정의
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
```

아래에서 `sentences`와 `labels` 변수에 `text`와 `category` 컬럼을 리스트 변환하여 담은 뒤 간단한 전처리를 수행합니다.



```python
# sentence 와 labals를 분리 합니다.
sentences = df['text'].tolist()
labels = df['category'].tolist()
```

`cleaned_sentences` 라는 빈 리스트를 생성한 뒤, 각 문장에서 불용어를 제외한 뒤 다시 쪼개진 단어를 합쳐서 추가합니다.



**불용어(stopwords) 란?**



문장 내에서 빈번하게 발생하여 의미를 부여하기 어려운 단어들을 의미합니다. 



‘a’, ‘the’, 'in' 같은 단어들은 모든 구문에 **빈번히 등장하지만 의미가 없습니다.**



특히 불용어는 자연어 처리에 있어 효율성을 감소시키기 때문에 가능하다면 제거하는 것이 좋습니다.



```python
cleaned_sentences = []

for sentence in sentences:
    # list comprehension
    cleaned = [word for word in sentence.split() if word not in stopwords]
    cleaned_sentences.append(' '.join(cleaned))
    
# 불용어 처리 전
print(f'[불용어 처리 전] {sentences[0][:100]}')

# 불용어 처리 후
print(f'[불용어 처리 후] {cleaned_sentences[0][:100]}')
```

<pre>
[불용어 처리 전] tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital
[불용어 처리 후] tv future hands viewers home theatre systems plasma high-definition tvs digital video recorders movi
</pre>
`train` / `validation` 셋으로 나눕니다.



```python
train_sentences = cleaned_sentences[:training_size]
validation_sentences = cleaned_sentences[training_size:]

train_labels = labels[:training_size]
validation_labels = labels[training_size:]
```

## 토크나이저 정의


`tensorflow.keras.preprocessing.text.Tokenizer`를 생성합니다.



- `num_words`: 몇 개의 단어 사전을 활용할지 지정합니다.

- `oov_token`: Out of Vocab 토큰을 지정합니다. 보통 겹치지 않는(일반적으로 사용하지 않은 특수 문자 조합으로..) 문자열로 지정합니다.



```python
# vocab_size = 1000
# oov_token 지정
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# 단어 사전을 생성합니다.
tokenizer.fit_on_texts(train_sentences)
```


```python
# 문장을 시퀀스로 변환합니다.
train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
```


```python
# padding 처리를 수행하여 한 문장의 길이를 맞춥니다.
# maxlen은 120 단어로 지정하였습니다.
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
```


```python
# 결과물 shape 확인
train_padded.shape
```

<pre>
(2000, 120)
</pre>

```python
# 0번째 index 출력
train_padded[0]
```

<pre>
array([101, 176,   1,   1,  54,   1, 782,   1,  95,   1,   1, 143, 188,
         1,   1,   1,   1,  47,   9, 934, 101,   4,   1, 371,  87,  23,
        17, 144,   1,   1,   1, 588, 454,   1,  71,   1,   1,   1,  10,
       834,   4, 800,  12, 869,   1,  11, 643,   1,   1, 412,   4,   1,
         1, 775,  54, 559,   1,   1,   1, 148, 303, 128,   1, 801,   1,
         1, 599,  12,   1,   1, 834,   1, 143, 354, 188,   1,   1,   1,
        42,  68,   1,  31,  11,   2,   1,  22,   2,   1, 138, 439,   9,
       146,   1,  80,   1, 471,   1, 101,   1,  86,   1,  93,   1,  61,
         1, 101,   8,   1, 644,  95,   1, 101,   1, 139, 164, 469,  11,
         1,  46,  56], dtype=int32)
</pre>
`1`로 마킹되어 있는 단어들이 많이 보입니다. `1` 로 마킹된 단어들은 **OOV 토큰**입니다.



```python
# label을 numpy array 로 변환합니다.
train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)
```

## 모델



```python
# model 생성
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(6, activation='softmax')
])
```

컴파일시 `loss`는 `sparse_categorical_crossentropy`를 지정하였습니다 (별도의 원핫인코딩을 수행하지 않았기 때문).



```python
# model compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```

체크포인트를 생성합니다. val_loss 기준으로 가장 최저점의 체크포인트를 학습이 완료된 뒤 로드합니다.



```python
checkpoint_path = 'bbc_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss',
                             verbose=1)
```

모델을 학습합니다. `epochs`를 충분히 주어 원하는 `val_loss`, `val_acc`에 도달할 때까지 학습합니다.



```python
history = model.fit(train_padded, train_labels,
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=30)
```

<pre>
Epoch 1/30
61/63 [============================>.] - ETA: 0s - loss: 1.6741 - acc: 0.2039
Epoch 00001: val_loss improved from inf to 1.63404, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 5s 33ms/step - loss: 1.6717 - acc: 0.2075 - val_loss: 1.6340 - val_acc: 0.1911
Epoch 2/30
61/63 [============================>.] - ETA: 0s - loss: 1.4633 - acc: 0.3294
Epoch 00002: val_loss improved from 1.63404 to 1.37903, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 1.4598 - acc: 0.3315 - val_loss: 1.3790 - val_acc: 0.3333
Epoch 3/30
61/63 [============================>.] - ETA: 0s - loss: 1.2494 - acc: 0.4252
Epoch 00003: val_loss improved from 1.37903 to 1.16279, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 1.2487 - acc: 0.4250 - val_loss: 1.1628 - val_acc: 0.4444
Epoch 4/30
61/63 [============================>.] - ETA: 0s - loss: 1.0220 - acc: 0.5277
Epoch 00004: val_loss improved from 1.16279 to 1.02034, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 1.0172 - acc: 0.5305 - val_loss: 1.0203 - val_acc: 0.5111
Epoch 5/30
61/63 [============================>.] - ETA: 0s - loss: 0.9038 - acc: 0.5758
Epoch 00005: val_loss did not improve from 1.02034
63/63 [==============================] - 1s 20ms/step - loss: 0.8989 - acc: 0.5805 - val_loss: 1.0432 - val_acc: 0.5422
Epoch 6/30
62/63 [============================>.] - ETA: 0s - loss: 0.7046 - acc: 0.7122
Epoch 00006: val_loss improved from 1.02034 to 0.72509, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 0.7021 - acc: 0.7140 - val_loss: 0.7251 - val_acc: 0.6756
Epoch 7/30
61/63 [============================>.] - ETA: 0s - loss: 0.5501 - acc: 0.7859
Epoch 00007: val_loss improved from 0.72509 to 0.70795, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 0.5455 - acc: 0.7890 - val_loss: 0.7079 - val_acc: 0.6711
Epoch 8/30
62/63 [============================>.] - ETA: 0s - loss: 0.3505 - acc: 0.8720
Epoch 00008: val_loss improved from 0.70795 to 0.45064, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 0.3484 - acc: 0.8730 - val_loss: 0.4506 - val_acc: 0.8311
Epoch 9/30
61/63 [============================>.] - ETA: 0s - loss: 0.2467 - acc: 0.9180
Epoch 00009: val_loss did not improve from 0.45064
63/63 [==============================] - 1s 20ms/step - loss: 0.2440 - acc: 0.9185 - val_loss: 0.4995 - val_acc: 0.8533
Epoch 10/30
61/63 [============================>.] - ETA: 0s - loss: 0.2481 - acc: 0.9191
Epoch 00010: val_loss improved from 0.45064 to 0.43606, saving model to bbc_checkpoint.ckpt
63/63 [==============================] - 1s 20ms/step - loss: 0.2450 - acc: 0.9200 - val_loss: 0.4361 - val_acc: 0.8844
Epoch 11/30
62/63 [============================>.] - ETA: 0s - loss: 0.1306 - acc: 0.9637
Epoch 00011: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.1298 - acc: 0.9640 - val_loss: 0.5338 - val_acc: 0.8667
Epoch 12/30
61/63 [============================>.] - ETA: 0s - loss: 0.1060 - acc: 0.9708
Epoch 00012: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.1040 - acc: 0.9715 - val_loss: 0.4474 - val_acc: 0.8844
Epoch 13/30
61/63 [============================>.] - ETA: 0s - loss: 0.1686 - acc: 0.9426
Epoch 00013: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.1688 - acc: 0.9425 - val_loss: 0.7164 - val_acc: 0.8044
Epoch 14/30
62/63 [============================>.] - ETA: 0s - loss: 0.1827 - acc: 0.9471
Epoch 00014: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.1816 - acc: 0.9475 - val_loss: 0.5031 - val_acc: 0.8667
Epoch 15/30
61/63 [============================>.] - ETA: 0s - loss: 0.0930 - acc: 0.9708
Epoch 00015: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0920 - acc: 0.9710 - val_loss: 0.5199 - val_acc: 0.8622
Epoch 16/30
61/63 [============================>.] - ETA: 0s - loss: 0.0404 - acc: 0.9898
Epoch 00016: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0442 - acc: 0.9890 - val_loss: 0.4765 - val_acc: 0.9022
Epoch 17/30
61/63 [============================>.] - ETA: 0s - loss: 0.0504 - acc: 0.9836
Epoch 00017: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0494 - acc: 0.9840 - val_loss: 0.5070 - val_acc: 0.8889
Epoch 18/30
62/63 [============================>.] - ETA: 0s - loss: 0.0448 - acc: 0.9854
Epoch 00018: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0444 - acc: 0.9855 - val_loss: 0.5073 - val_acc: 0.8844
Epoch 19/30
61/63 [============================>.] - ETA: 0s - loss: 0.0404 - acc: 0.9867
Epoch 00019: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0405 - acc: 0.9865 - val_loss: 0.7229 - val_acc: 0.8578
Epoch 20/30
61/63 [============================>.] - ETA: 0s - loss: 0.0649 - acc: 0.9790
Epoch 00020: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0636 - acc: 0.9795 - val_loss: 0.5101 - val_acc: 0.9022
Epoch 21/30
61/63 [============================>.] - ETA: 0s - loss: 0.0333 - acc: 0.9898
Epoch 00021: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0328 - acc: 0.9900 - val_loss: 0.6735 - val_acc: 0.8844
Epoch 22/30
62/63 [============================>.] - ETA: 0s - loss: 0.0218 - acc: 0.9934
Epoch 00022: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0217 - acc: 0.9935 - val_loss: 0.5494 - val_acc: 0.9067
Epoch 23/30
61/63 [============================>.] - ETA: 0s - loss: 0.0375 - acc: 0.9887
Epoch 00023: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0378 - acc: 0.9885 - val_loss: 0.5719 - val_acc: 0.9111
Epoch 24/30
61/63 [============================>.] - ETA: 0s - loss: 0.0800 - acc: 0.9734
Epoch 00024: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0804 - acc: 0.9730 - val_loss: 0.6355 - val_acc: 0.8844
Epoch 25/30
61/63 [============================>.] - ETA: 0s - loss: 0.0606 - acc: 0.9872
Epoch 00025: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0592 - acc: 0.9875 - val_loss: 0.5286 - val_acc: 0.9067
Epoch 26/30
61/63 [============================>.] - ETA: 0s - loss: 0.0158 - acc: 0.9954
Epoch 00026: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0162 - acc: 0.9950 - val_loss: 0.5403 - val_acc: 0.9022
Epoch 27/30
61/63 [============================>.] - ETA: 0s - loss: 0.0128 - acc: 0.9974
Epoch 00027: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0125 - acc: 0.9975 - val_loss: 0.5612 - val_acc: 0.9022
Epoch 28/30
61/63 [============================>.] - ETA: 0s - loss: 0.0074 - acc: 0.9990
Epoch 00028: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 20ms/step - loss: 0.0073 - acc: 0.9990 - val_loss: 0.5918 - val_acc: 0.9022
Epoch 29/30
62/63 [============================>.] - ETA: 0s - loss: 0.0087 - acc: 0.9990
Epoch 00029: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 19ms/step - loss: 0.0087 - acc: 0.9990 - val_loss: 0.6322 - val_acc: 0.8978
Epoch 30/30
62/63 [============================>.] - ETA: 0s - loss: 0.0070 - acc: 0.9985
Epoch 00030: val_loss did not improve from 0.43606
63/63 [==============================] - 1s 19ms/step - loss: 0.0070 - acc: 0.9985 - val_loss: 0.6204 - val_acc: 0.9022
</pre>
학습이 완료된 뒤 저장한 체크포인트를 load 합니다.



```python
# checkpoint 로드
model.load_weights(checkpoint_path)
```



`validation_padded`와 `validation_labels`로 최종 성능평가를 수행합니다.



```python
# 모델 평가
model.evaluate(validation_padded, validation_labels)
```

<pre>
8/8 [==============================] - 0s 8ms/step - loss: 0.4361 - acc: 0.8844
</pre>
<pre>
[0.43605977296829224, 0.8844444155693054]
</pre>
## 학습 결과 시각화



```python
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2)
fig.set_size_inches(10, 4)
axes[0].plot(history.history['loss'], color='#5A98BF', alpha=0.5, linestyle=':', label='loss')
axes[0].plot(history.history['val_loss'], color='#5A98BF', linestyle='-', label='val_loss')
axes[0].set_xlabel('Epochs', fontsize=10)
axes[0].set_ylabel('Loss', fontsize=10)
axes[0].set_title('Losses')
axes[0].tick_params(axis='both', which='major', labelsize=8)
axes[0].tick_params(axis='both', which='minor', labelsize=6)
axes[0].legend()

axes[1].plot(history.history['acc'], color='#F2294E', alpha=0.3, linestyle=':', label='acc')
axes[1].plot(history.history['val_acc'], color='#F2294E', linestyle='-', label='val_acc')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy')
axes[1].tick_params(axis='both', which='major', labelsize=8)
axes[1].tick_params(axis='both', which='minor', labelsize=6)
axes[1].legend()

plt.show()
```

<img src="data:image/png;base64,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o0eP7q2nIkS7JJkSYgCbM2cOmZmZ/P3vf2fixIm89NJLpKSk8NJLL7nG3Hnnnezfv5/Vq1fj6enJgw8+yPjx413f7GmaBjj3TX3yySdtbv/+9787fS0hhBADy+eff05hYSFvvPEGY8aMcd0uueQSgF6v6tfRDFXLDNjRzGYzqqq2GXvWWWfxwQcfcN999/Hee+/xySefuIpXtLxvdcU111zDV199RX5+PpmZmXz//fcyKyX6hRSgEKKfhYWF4e3tzb59+9rct3fvXlRVJTY21nUsODiY6667juuuu466ujrmzJnDww8/zLJly1xjEhMTueeee7jnnns4cOAAU6dO5emnn+bVV191bRgODw/nzDPPPG58x7qWEEKIgWX9+vWEh4fz17/+tc1977zzDu+++y4vvPACiYmJ7N69+5jXSkxM5IcffsBms2EymdodExQUBDgrAx4pJyen0zHv2rWL/fv3849//INrrrnGdfzISrWAayn88eIGuOyyy7j77rt5/fXXaWxsxGQycemll3Y6JiG6S2amhOhnBoOBs88+m3//+9+tSo4XFxfz2muvMXv2bPz9/QEoLy9vda6vry+jR4/GYrEA0NDQQFNTU6sxiYmJ+Pn5ucbMnz8ff39/nnjiCWw2W5t4SktLO30tIYQQA0djYyPvvPMOF1xwAUuWLGlzu/XWW6mtrWXjxo0sXryYHTt2tFtCXNd1wFnRtaysjGeffbbDMSNHjsRgMPD111+3uv+5557rdNwGg6HVNVv+/Kc//anVuLCwMObMmcPf//53cnNz242nRWhoKOeeey6vvvoq69ev55xzziE0NLTTMQnRXTIzJUQf+vvf/85///vfNscffvhhPvnkE2bPns0vfvELjEYja9aswWKx8NRTT7nGJScnM3fuXKZPn05wcDBbtmxhw4YN3HrrrQDs37+fM844g0suuYTk5GSMRiPvvvsuxcXFXHbZZQD4+/vz/PPPc/XVV5OSksJll11GWFgYubm5fPDBB5xyyik8++yznbqWEEKIgWPjxo3U1tayYMGCdu8/6aSTXA18X3vtNTZs2MDSpUu5/vrrmT59OhUVFWzcuJEXXniBKVOmcM011/DKK69w99138+OPP3LqqadSX1/Pp59+yi9+8QsuuugiAgICWLp0KX/5y19QFIXExETef/991/7bzhg3bhyJiYn88pe/5NChQ/j7+/P222+3W/zoz3/+M7NnzyYlJYXly5czatQosrOz+eCDD9i+fXursddccw1LliwB4NFHH+38CylET7izlKAQQ1VLafSObnl5efrWrVv1+fPn676+vrq3t7d++umn6999912r6zz22GP6iSeeqAcGBupeXl76uHHj9Mcff1y3Wq26rut6WVmZfsstt+jjxo3TfXx89ICAAH3mzJn6v/71rzYxffHFF/r8+fP1gIAA3dPTU09MTNRTU1P1LVu2dPlaQggh3O/CCy/UPT099fr6+g7HpKam6iaTSS8rK9PLy8v1W2+9VR8xYoTu4eGhx8TE6Ndee61eVlbmGt/Q0KA/8MAD+qhRo3STyaRHRkbqS5YsadXKo7S0VF+8eLHu7e2tBwUF6StWrNB3797dbml0Hx+fduNKT0/XzzzzTN3X11cPDQ3Vb7zxRn3Hjh1trqHrur5792794osv1gMDA3VPT089KSlJf/DBB9tc02Kx6EFBQXpAQIDe2NjYyVdRiJ5RdP2oeVIhhBBCCCEGGbvdTnR0NBdeeCFr1651dzhimJA9U0IIIYQQYtB77733KC0tbVXUQoi+JjNTQgghhBBi0Prhhx/YuXMnjz76KKGhoWzdutXdIYlhRGamhBBCCCHEoPX8889z8803Ex4eziuvvOLucMQwIzNTQgghhBBCCNENbp2Zuv3224mPj0dRlDblLVusW7eOqVOnum6hoaEsWrQIgOzsbAwGQ6v7MzMz+/EZCCGEEEIIIYYrt85Mff311yQkJDB79mzee+89pk6detxzJk6cyKpVq1i8eDHZ2dlMnTq1TRduIYQQQgghhOhrbm3aO2fOnC6N/+GHHygpKemwOV1XaZpGQUEBfn5+KIrSK9cUQghxfLquU1tbS3R0NKoq23ePJO9NQgjhHt15b3JrMtVVa9eu5eqrr8ZkMrmO1dfXM2PGDBwOBwsXLuSBBx7AYDC0e77FYsFisbh+PnToEMnJyX0etxBCiPbl5eURExPj7jAGlIKCAmJjY90dhhBCDFtdeW8aNMlUfX09b7zxBt9//73rWFRUFIcOHSI8PJyKigouvfRSnn76ae699952r7F69WpWrVrV5nheXh7+/v59FrsQQojWampqiI2Nxc/Pz92hDDgtr4m8NwkhRP/qznvToEmm3nrrLSZMmNBqJslsNhMeHg5AcHAw119/Pa+99lqHydTKlSu5++67XT+3vGD+/v7yhiWEEG4gy9jaanlN5L1JCCHcoyvvTYMmmVq7di033HBDq2MlJSUEBQVhMpmwWCy88847TJs2rcNrmM1mzGZzX4cqhBBCCCGEGAbcuut3xYoVxMTEkJ+fz/z58xk9ejQAy5YtY+PGja5x+/btY/v27Vx66aWtzv/mm2+YNm0aU6ZMISUlhcjISB544IF+fQ5CCCGEEEKI4WlYN+2tqakhICCA6upqWUohhBD9SH7/dkxeGyGEcI/u/P4dNMv8hBCiI7quY7fbcTgc7g5FHMFkMnVYXVUIIYQYCiSZEkIMalarlcLCQhoaGtwdijiKoijExMTg6+vr7lCEEEKIPiHJlBBi0NI0jaysLAwGA9HR0Xh4eEh1uAFC13VKS0vJz89nzJgxMkMlhBBiSJJkSggxaFmtVjRNIzY2Fm9vb3eHI44SFhZGdnY2NptNkikhhBBDklur+QkhRG9QVflVNhANxVnC22+/nfj4eBRFYfv27R2OW7t2LWPGjCExMZEbb7wRm83Wf0EKIYToN/IJRAghhOikJUuW8M033zBy5MgOx2RlZfHggw+yadMmMjIyKC4u5sUXX+zHKIUQQvQXSaaEEEKITpozZw4xMTHHHLNhwwYWLFhAZGQkiqJw00038frrr3c43mKxUFNT0+omhBBDga5pHNmFSbdY0Spr0OtaF43SyqvQSivQbfbDY5sszmNVta2vWV2HXlXTeqzNjl7fgN5o6aNn0jFJpnqgwWIns6QGi03KMQshumbu3Lnceeed7g5D9IHc3NxWM1fx8fHk5uZ2OH716tUEBAS4brGxsf0RphBCdJleVYtWWomuaa5jWmkljp370XILW421f7MVx1db4IgERy+rRNu+Fy2noNVYbV822u4MaDpibFUt2u6MNmMdew/i2LYX6huOGFuD48fdOPYc7JXn2RWSTPXAd5nFbM8tp7BaSjILIYTonpUrV1JdXe265eXluTskIUQv0o/qgajX1qM3NLWasRlotIpqHOmZaPnFrY47duxD230ALEfsA7VY0cur0GvqWl+kZT/zEYkXHibw8QIPj1ZDFT8fCPA7fA6Ayeg85u3VeqyXJ3h7wpGFjRQVPEwoxv4vdiTV/HpgRKAPJlXFww3/4YQQQgxMcXFxZGZmun7Ozs4mLi6uw/Fmsxmz2dwfoQkh+ohWXgVNFpSwYBQPk/NYURnavmyU4AAMk8a4xjr2ZkFdA+rksSghgQDojRb0mjoUHy8U3/6tTqtX1aJX16FEh6GYmlODxib04nKw2SEm4vBgPx9ncnREIqgE+aEmjULx9mx1XcP0ZGdydMTnZDUsGDUsuE0MR74+rrEhgajNr0+rsZPHth0bGogaOu14T7VPyMxUD4yLCuC0cVFEB0pJZiEGGrtDw+5ovVZb03TsDg3Hkd+S9cLYnqqsrOSaa64hKCgIb29vzj33XA4cOOC6PycnhwsvvJCgoCB8fHyYMGECH374oevcK6+8krCwMLy8vBgzZgzr1q3rtdhE1y1evJiNGzdSVFSEruu88MILXHbZZe4OS4gBQcvKRysoRbcP3i0Suq6jW6ytjmkHctH250BD4+GDBgNoWpuxqCqoKsoRMy56ZQ1aeiZaZuuZab22vtWSul6Jvb6x1THH3iy0g3mtZpaUoADUUSNQR0a3GmtMGY/xhAmtEifFxxs1Ogwl0K/VWMXsgWIyDsnKrkeSmakeGOp/OYQYzP69LQeA86fE4Wlyfiu2r6ia9IJK4kP9mB4f6hr7/o48HJrGOZNi8DE7v1HMLK1hZ14FscG+nJgQ5hr70a58rHYHZ04YQYBX62UK3ZWamsqBAwfYuHEj/v7+3HfffZx33nmkp6djMpm45ZZbsFqtfP311/j4+JCeno6vry8ADz74IOnp6Xz00UeEhoaSkZFBY2PjcR5RdNeKFSv44IMPKCoqYv78+fj5+ZGRkcGyZctYsGABCxYsICEhgVWrVnHKKacAzv1xK1ascHPkQvQvXdedMxu6jhrl/B2q2+xo2c79L4bww7MTWnE5emUNSliQayZCdzigrhF0DSXQ//DYsiqorUMJCnB9eNetNrTmvTKGKUmusY6D+ehFZagxEahxUYePZ+Si+HqjhAWhdLEHnl5dhyMtA0xGjDMmuo4rQf7O5WtHLFNTgvwxnDQFmt9XWhinJ6PreuvPkSYDSqAfir/P4ceyO3Ds2AeKgmHqOBSf1svdukqva3DGroPxpMmH4wwLgsamVq+F4u2JEj+iR483XEgy1QO6rlPdaMXf04N6iw2/XvpgJYQYPlqSqG+//ZZZs2YBsH79emJjY3nvvfdYunQpubm5LF68mEmTJgGQkJDgOj83N5dp06ZxwgknAM5iB6LvrFmzpt3jL730Uqufb7zxRm688cb+CEmIAUkvrXAmOEYjSmiQc/mYrqPERIDV1mpvi15ejV5chuJphpZlXRYbjq3pYDBgnDP98NiySvTCUlRFbTUToldUtw3C4XDu5zliFkxvsqDnFaErCoawoMPHq2rRNQ3F39cVm67rUF0LBoNzTw+AlxmsNrA70K0215I+Q1J8m4dXjIZWS9xa3XfUF/JqWDAcvfytsck1i8VRS+i6xewBTVZQVXS7w/U8DYlS9KYnJJnqpop6Cw+/t5W6JhsLpsZhNKicNyUWVWarhBgQLprmrKZmUA//m0yKDGBMhD9H/zO9YEpsm7GJYf6MCvVrM/bcSTFtxvbEnj17MBqNzJw503UsJCSEpKQk9uzZAzgbxd588818/PHHnHnmmSxevJjJk53fKt58880sXryYrVu3cvbZZ7Nw4UJXUiaEEP3pyNkWJSwYAkqcM00G52yN4mHCMKZtjzYlMgQ8PZyzOy2MBvA0t0lGlEA/UBXw8241Vh03ypkkHBGDGhsJkaHOJOLIa8REgsPRaiZGyytCL6tEHR2HEhvpPJZ1CD2nACUiBENy4uHnMHUc+Pug9HHDeMXPB8NJk6HJ2ir5cuzYh+LvgxIb1WHBBd1idVbXszswjHd+AaeYjBimjAVfH7cUahiqZM9UNwV6e2C1a1jsGhX1zjKODRb7cc4SQvQXo0HFaFBbvQGpqoLRoGI46g2wp2P72rJlyzh48CBXX301u3bt4oQTTuAvf/kLAOeeey45OTncddddFBQUcMYZZ/DLX/6y32ITQgjdZseRkYu2fa9rP6miKBhTxqOOjDpu0qEGB2BIiEEJ8HUdUzxMGE+e0mopHYAaGYphbHyrwgSKqqJGhaFGhLT63ax4mlH8fFyzRy3HDGPiMIwb1ToITw9oHu96rJBAMBqdVeWOoAT69Xki5XosVW21P0mrqEavqEbLLWpdJe9oNjt6fjF6cTn6EeXGlUB/SaR6mSRT3aQqCsnRgQD4mE2cNyUWX0/TsU8SQoijjB8/Hrvdzg8//OA6Vl5ezr59+0hOTnYdi42N5aabbuKdd97hnnvu4W9/+5vrvrCwMK699lpeffVVnnnmGV588cV+fQ5CiGFO09APlTiXylUOzqbThjEjMZ48pdXSQSXAF8MpU9udTXMXJcgfdUIiamJsqyRRyylEKyw9PM7XGzU+GnVKknP5pOgzssyvB5JHBPLDwVIySmpkeZ8QolvGjBnDRRddxI033siaNWvw8/Pj17/+NSNGjOCiiy4C4M477+Tcc89l7NixVFZW8sUXXzB+/HgAHnroIaZPn86ECROwWCy8//77rvuEEEOPrmloe7PAoaGOiXPbB+Uj9wspZg/UMSPBbEINDnBLPH2lv2agOktRFJTwkFbH9IYmtKx8MBpQwoNdyxfVUTHuCHHYGVh/QwaZCdHOjYsHiquxNm9uHMgN2IQQA9O6deuYPn06F1xwASeffDK6rvPhhx9iMjk/qDgcDm655RbGjx/POeecw9ixY3nuuecA8PDwYOXKlUyePJk5c+ZgMBh444033Pl0hBC9RG+0OGccjmicqqgqenWds0lqXf9X7tR1HS37EI7vd6LXN7iOq9Fh7fYEEv1A05xJVFQYaPI5tL8p+jD+9F9TU0NAQADV1dX4+/sf/4Sj6LrObes3U1lvZdlpY9E1GBHkTVJUYO8HK4Roo6mpiaysLEaNGoWnZy9UOhK96lj/fXr6+3cok9dmeNI1Db2iBsXPG6W5YIJWWom2+wB4e2KcebiUtVbgXM6lRoe1e60+jVPX0XbuR6+odvYhkvLZYgjpzu9fmZnqAUVRSG6ZnSqqobLBQmG19HcRQgghROfpmoZj6x60XfvRy6pcx5UAX5TQINTo8Fbj1eiwVomUbrOj9dNeJUVRUMcnoCYnSiIlBJJM9diE5iIU+ZX1pIwM5aTE/v+WSAghhBCDl6KqqCPCnZXjjqxG52HCMGmMs8R3B3RNQ9t9AG3HPrTi8l6PTdc0Z6W+5ma7LXGpESHHOEuI4UMKUPRQ8gjnzFRWaS2RAV54muQlFUIIIUTXqFFhKBEh3St44GmG2gYUH69ej0svr3Y1uVUiQlC8pDKcEEeSmakeCvPzJNzfE02HfUXtdN8WQgghhGiHbrOjH9ErqDuJlKKqGMYnYDhhAoqv9/FP6CI1LAglJgJ1wmhJpIRohyRTvaBl31TaoUryK+vZklWKzXGMRmpCCCGEGNZ0XUdLz8Txczp6Q1OPr3dkY1e9oQnHgZxWiVqn43I40LILWp1rGDMSNSyoxzEKMRRJMtULJowIBCC9oIrd+ZXklNdRXCOFKIQQQgjRgUYLem09NDRBLxZW1jUNx8796PnFaJn5XT5f23UALSsfLTOv12ISYiiTZKoXJDcXocgpqyMqwIsxEQH4e5qOfZIQQgghhi3F2xPDjInO5XO9uNdJUVXUMXHg6406MqrVfXpzT8wWjvRM7N9tdyZ1LefHRoKHCUV6RgnRKVItoRcEepsZEeTNocoGrA6NGaOkop8QQgghjk0xe7h6SvUmNSQQJTgApbkyoFZRjbY7A3y9MaaMPzzQYgWLFb2hCcXP5/C5J01GMRh6PS4hhiKZmeolE5r3TaUfqnJvIEIIIYQYsBwH89Fr6vr8cZSjSqzjcEBD6y0I6qgYDCnJbWahJJESovMkmeolLUv90goq0XWdmkYrxdLAVwjRR+Lj43nmmWc6NVZRFN57770+jUcIcXxaSTl6TgGObXvRLdb+e2BvTwwnTsQwa2qrw0qgn7MxsFGSJyG6S5b59ZJx0YEowKHKBg4UV7MrvxJvDyPnTIpp9e2QEEIIIYYnJSgAJTwYxcerT5b3dfi4qgo+vV82XQghM1O9xs/TxMhQXwCKa5owGVT8vTykRLoQQgghAFBMRgwTRqOMjHZ3KEKIXiLJVC9qWeq3t7CaC6bEccqYCDxk6lyIfqXrOk02R7/f9C6UNn7xxReJjo5GO6oHzEUXXcT1119PZmYmF110EREREfj6+jJjxgw+/fTTXnuNdu3axbx58/Dy8iIkJITly5dTV3d4D8eXX37JiSeeiI+PD4GBgZxyyink5OQAsGPHDk4//XT8/Pzw9/dn+vTpbNmypddiE2IoOrqPlKxYEWLokGV+vWjCiCA+3JlPekElqiq/KIVwB4td44a/b+r3x117/al4mjr35cnSpUu57bbb+OKLLzjjjDMAqKio4L///S8ffvghdXV1nHfeeTz++OOYzWZeeeUVLrzwQvbt20dcXFyP4qyvr2f+/PmcfPLJ/PTTT5SUlLBs2TJuvfVWXn75Zex2OwsXLuTGG2/k9ddfx2q18uOPP7o+/F155ZVMmzaN559/HoPBwPbt2zGZpBWEEB3RKmvQduxDiYlATYyVREqIIcatM1O333478fHxKIrC9u3b2x3z5Zdf4uXlxdSpU123xsbDhR3Wrl3LmDFjSExM5MYbb8Rms/VT9G0lRQagKlBS00RprfNbKLtDwy5L/YQQRwgKCuLcc8/ltddecx3bsGEDoaGhnH766UyZMoUVK1YwceJExowZw6OPPkpiYiIbN27s8WO/9tprNDU18corrzBx4kTmzZvHs88+yz//+U+Ki4upqamhurqaCy64gMTERMaPH8+1117rSuJyc3M588wzGTduHGPGjGHp0qVMmTKlx3EJMWRV1Tib8todkkgJMQS5dWZqyZIl3HvvvcyePfuY45KSktpNtrKysnjwwQfZunUrERERXHTRRbz44ovccsstfRTxsXl5GEkM9+dAcQ3pBZWE+HhyoKSaSSOCGR3h75aYhBhuzEaVtdef6pbH7Yorr7ySG2+8keeeew6z2cz69eu57LLLUFWVuro6Hn74YT744AMKCwux2+00NjaSm5vb4zj37NnDlClT8PHxcR075ZRT0DSNffv2MWfOHFJTU5k/fz5nnXUWZ555JpdccglRUc7mn3fffTfLli3jn//8J2eeeSZLly4lMTGxx3EJMdjoVTVo5dUovt6oESGu447te9FtdgyTx6KYPVBHxYC/L0qgnxujFUL0FbfOTM2ZM4eYmJhun79hwwYWLFhAZGQkiqJw00038frrr/dihF3Xsm8q/VAVHkYVTdOpqLe4NSYhhhNFUfA0Gfr91tVvnC+88EJ0XeeDDz4gLy+PTZs2ceWVVwLwy1/+knfffZcnnniCTZs2sX37diZNmoTV2j+llNetW8fmzZuZNWsWb775JmPHjuX7778H4OGHHyYtLY3zzz+fzz//nOTkZN59991+iUuIgUSvbUDPLUQvr2p9vK4B6hrAbncdU0MCpXeTEEPUoChAkZmZSUpKCjNmzOC5555zHc/NzWXkyJGun+Pj44/5za3FYqGmpqbVrbdNGOFs3ptWUElssDdnJo9gxqjQXn8cIcTg5unpyaJFi1i/fj2vv/46SUlJpKSkAPDtt9+SmprKxRdfzKRJk4iMjCQ7O7tXHnf8+PHs2LGD+vp617Fvv/0WVVVJSkpyHZs2bRorV67ku+++Y+LEia2WJI4dO5a77rqLjz/+mEWLFrFu3bpeiU2IgUzXdfSmw1+OKn4+KDGRbRrequMTUKckgdnczxEKIdxhwBegSElJIT8/n4CAAPLz8znvvPMIDQ3lkksu6fK1Vq9ezapVq/ogysPGRPhjVBUq661UNdiICpS+DkKI9l155ZVccMEFpKWlcdVVV7mOjxkzhnfeeYcLL7wQRVF48MEH21T+68lj/va3v+Xaa6/l4YcfprS0lNtuu42rr76aiIgIsrKyePHFF1mwYAHR0dHs27ePAwcOcM0119DY2MivfvUrlixZwqhRo8jPz+enn35i8eLFvRKbEAOZnlOIlluIOnYkamQoSqAfhnaW7qlHJVdCHI9us6MfKkHLLUTLK3L+f24hemklxjNPwmP5EmevMDEgDfhkyt//8F6jmJgYLr/8cjZt2sQll1xCXFwcmZmZrvuzs7OPWelq5cqV3H333a6fa2pqiI2N7dV4PYwGxkQGsKegirSCSkmmhBAdmjdvHsHBwezbt48rrrjCdfwPf/gD119/PbNmzSI0NJT77ruv12bSvb29+d///scdd9zBjBkz8Pb2ZvHixfzhD39w3b93717+8Y9/UF5eTlRUFLfccgsrVqzAbrdTXl7ONddcQ3FxMaGhoSxatKjPv6QSwt10XUevqgGHw92hiEFM1zQc3+9E25/jSpi0vCL0ghLooFiZNT0Tbc9BPFffieLt2StxONIz0RstqHFRKKGBUhilhwZ8MlVYWEhERASqqlJbW8v777/PDTfcAMDixYuZPXs2Dz/8MBEREbzwwgtcdtllHV7LbDZj7odp9wnRgewpqCL9UBVzk6LYU1hFaU0Tp42LxCDfLAghmqmqSkFBQZvj8fHxfP75562OHV1YpyvL/o7ugTVp0qQ2128RERHR4R4oDw8Pt+9LdbcDBw5w7bXXUlZWRkBAAC+//DITJkxoNUbTNO69917++9//YrfbOeWUU3j++efx8PBwU9SipxRFQZ2ShF5WiRoW7O5wxCCkFZXR9MCfcWze0f4ALzNqbKQzwYmLQo2NhCYLlj+8gv2TzTTkF+P11wdQI7u/dUSvrqPpsTXYP/j6iMf1RI2LQo1r/dhqXBRKZIjMiHWCW5OpFStW8MEHH1BUVMT8+fPx8/MjIyODZcuWsWDBAhYsWMDbb7/N888/j9FoxG63s3TpUq677joAEhISWLVqFaeccgoAc+fOZcWKFe58SgAkjwiCLdmkF1ShKJBbXk+TzU5pbRORATJTJYQQg9WKFStYvnw5qampbNiwgdTUVH766adWY9auXcvWrVvZunUrJpOJ5cuX86c//Ylf/epXbopa9AZFUVAkkeoW+3fbsb6yEeNpJ2BaMBfFZ2B/FnIczMf6x1cwnDwF09L5KKaefVy2fbSJplXPQ009eHpgnDsDdWQ0amykM3kZGYUSGtTuDJE6aSxNt69G23OQhkt+idez92OYPLbLMdi/30nT/c+gF5WDqqJEhqIXlUFjE9q+LLR9WW1P8jChxkSAn0/b+3qB4u/rTNxGRrkSOmVEBIpH53oX6hYrWn4x+tHLI0sq8H73T/0246boR39lOYzU1NQQEBBAdXV1q+WEPWV3aCx/+Rssdo3VS05A03UMqkJkgDcmg2T4QvSWpqYmsrKyGDVqFJ6evbP8YbBZv359h18ijRw5krS0tH6O6LBj/ffpq9+/famkpITRo0dTUVGB0WhE13WioqL45ptvGD16tGvcrbfeSnR0NPfffz8A77zzDg8//DA7d+5s97oWiwWL5XBhg5Yl6IPptRmqtPIqaGhCiYmQpVDd5MjIpeGyX0GDs/8mPl6YLpqH6fJzMST27laL3qA3WmhYejfawXwA1FEjMN9zLYbTT+zy3wG9po6mRw/PBKmTxuD15J3OcvldoB0qpvEXj6MdyAGzB55P3I7p3M61ANEtVizP/BPbP5x9CpXYSLyeuhvDlCR0qw39UDFa7uFERMstQssrRM8vBrsblrU2J3otM2XO2bJI0HEmSkfEqReVOXu4tcPnq3XdmkXuznvTgF/mNxgZDSpJUYHszKsg7VAl504eeL8shBBDw4IFC5g5c2a795lMnft2T3ROXl4eUVFRGI3Ot05FUYiLiyM3N7dVMjV9+nTWrFnDrbfeipeXF//617+OuSyzP4ojia7TbXa0vVlgtaEqCkpMhLtDGnT02noab1sNDU2o4xOgyYKWdQjbax9ge+0DDDMnY7riPIynn4hiHBil4y3/tw7tYD5KcAAAWtYhGm99AsOJEzHfez2G5M711Ws1E2RQ8Vi+FI+bLunWLJc6IgLv156k8Vd/wPHlTzTd83u0zHw8brnsmAmeY28WTff+AS3DWenatHQ+5nuvQ/HxAkDxMKGMimk3udPtDvSiMrTcQmjq/RY/ug56RTV6XnNylONM6GhsQi8owVFQguP79r+AasXHy5V0qbGRKM2zXIq/b6/H3BFJpvrIhGhnMpVeUCXJlBCiz/j5+eHnJ81AB5LU1FRycnI47bTT8PLy4swzz+Tjjz/ucHx/FEcSXaeYjKjxI9AKS1GipMVJV+maRuN9f0TPKUCJDMXrbw+jBPnj2LwD2+sfYv/iJxw/7MTxw06UyBBMl5yDacnZqKGBbovZ9tkP2N74CADPp+7GMHks1r+9jfUf/8bx424altyNccFczHdchRoV1u41nDNBr2L7x78BUOKi8PrdXRimJLU7vrMUH2+8/rISy9OvYHv5PazPvYF2MA/PJ+5A8WxdD0B3OLC+/G+sf1oPdjtKSACej96Gce6Mzj+e0YASE+Fc5tdPdF1HL68+PAOVV+iaNUNRXAlTy7JAJS4KJcjf7bPGssyvj5aZZJXW8pt3fsbLw8Caa0/B7tAprG5AVRTiQvovWxZiKGtZRhYfH4+Xl5e7wxFHaWxsJDs7e9gt8zvaG2+8wV//+lc2bdrUqccZjK/NUKbruts/rA1Glmdfx/rcG+Bhwnv9kxgmtP43ohWUYnvzv9je/gS9otp50GjEOH8WpkvPwTBpDIq5/4q2aCXlNCy8A72qFlPqQjzvva5VrJZn/on9/a+cB8weeKRehMeyRa32fzn2ZtF03x+dy/FoOxPUW6xvf4Jl1fNgdziXDv5lJWp4iDPWQ8U0rfwTji3OJd7G00/E/MgtUrK/k2SZ3wAyMsQXbw8jDVY72WV1mAwqP2eX4efpIcmUEL2kZRlbQ0ODJFMDkNVqBcBgGBjLd3oqPDyclJQUXn31VVJTU3n77beJiYlpk0g1NTXR2NhIUFAQZWVlPPnkkzz66KNuilp0lW61gdHgqmImiVTX2T//wZlIAZ6rftEmkQJQo8Mw33U1Hrdchv2/32J9/UO0Hfuwf/C1c4+RoqBEhLSuMueqNBfZq0UsdE2j6dfPoFfVoo5PwHznVa3uV6PD8HrqbhxXX4jlqb/j+Dkd65q3sG34BI/brsC0cB62f/4Hy5/Xg615JuiRWzGefmKvxXgkj8VnocZF0Xj7k2i7DtBw6a/w+usDaAdyaHrsRahvBC9PzPcvw7ToTPk73MdkZqoPv/37w/928XN2OZfNTGD+xBF8s7+YyEAvxkUFospfbCF6RWFhIVVVVYSHh+Pt7S1vGgOEpmkUFBRgMpmIi4tr899lsM6+7Nu3j9TUVMrLy/H392fdunVMmjSpVRXa4uJi5s6di6qqaJrGHXfcwU033dTpxxisr81QoOs6jm17QdMwTBiN4tWzdiq2Tzaj7c/BdPm5qM17cHoco6Zh/9+3aAWleFx70YDZa9TCcTCfhkt/CfWNmK48H88Hlnf+3LQMrK99iP3T76G2/phjlZBAlLhIDAkxmK67GENC14o6HMn693ex/P5l8DLjs+EPxywQoes69s++x/L7f6DnFjoP+vm44u3PmSAtt5DGXzzmLJZhUF29qtRp4/BafSdqXFSfxzDUdOf3ryRTffiG9d9d+fzzuwwmxQTx6/On9Pr1hRDON7aioiKqqqrcHYo4iqqqjBo1qt3+SpIwdExeG/fR6xtwbN0Luo5hxsQeJVPW9R9gefxF5w++3nisWIrHVRf0aOmafUsalqf+jrY7AwDzI7fgseTsbl+vt+l1DTRc9iu0g/kYpifj9fdHu1VwQdd19Mqa5sptRa0b3OYWolce1cTc1xuvP/wK4+yULj+WIz2ThsvuBbsd86pb8FjauddTt9qwvflfLM+9CdW1zpmglTdgWnxWv36pp9fW03jP/+H4ZhsYDXj84jI8li0ecEn2YCHJVBf19RtWXkUdv35rCx5Glb+lzsYoZdGF6DMOhwObzebuMMQRPDw8UDto+CgJQ8fktXEvvcmC3tDUo5kk68v/xvLU3wFQIkOcFd0AJToc891XYzz31C594NayD2F5+h/YP/vBeUBRQNdRJ43B583fdzvO3qRrGk13PIn9sx9QIkLwfusPfVZMQq+td/YVyinEtv4DHFvTQVUx//oGTFee3+nXVm9oon7JXejZBRjPOhnPZ+7revnz6jpsH3+H8eQp/VqsoVUMdgf2jzahJsVjGBvvlhiGCtkzNcDEBPng72mipslGRkkN46IC0XWdygYrvmYjHvKtgRC9xmAwDJm9OUII91E8zW2qo3WF5W8bsP7xnwB4LF+Cx21XYP/PV1j+9Cp6QQlNv3wa9R8bMd93A8aU8ce8ll5Vg+W5N50V5uwOMKiYls7HdPm5NCy+G23XARx7DmIYn9DteHuL9cUNzmTPZMTrmfv6tCqf4ueDITkRQ3IixjNm0vTwc9jf+xzLE39DO5iPeeWyTs2IWZ58CT27ACUiBM9Vt3RrRkkJ8O30bFZfUYwGTBfOdWsMw5lMlfQhRVFIHhEIQNqhKgC+yyjhiz0FHKpscF9gQgghhHDRcgrRj7NHpzMsz71xOJG69XI87rgKxWDAtHAePh8+j8ftV4KXJ9quAzRe9Wsa73zSWfb5KLrVhvXv71I3/yZsr74PdgeG007A+90/4fnQTRjGjMR4prO/nG1Dx2X3+4v9qy1Y//IaAOaHbupxGfCuUDxMeD5+O+Z7rgVFwfbGRzTe9Ah6dd0xz7N9/B22DZ+AouD55J0ogdJiQnSPJFN9bEJ0EADpBZUABPuYMRpUrM2bBIUQQgjhPrrFipaVj2PrHvRuNifVdR3Ln9djffZ1ADzuvBrzL1o3VFW8zJhvugSf/76AaenZoKrYP95M/QW30vTkS+hVtei6ju2jb6i/4BZnQYTaetSkUXitfQTv5x/EMDrOdT3T0vkA2P7zFXpDU/dfgB7ScgppvPcPoOuYLj0Hj8Vn9XsMiqLgccMivP6yErw8cWzeQf3lv0LLLmh3vFZYStNDfwXA44ZFGGdO7s9wxRAjy/z6WMvM1IHiGiw2B6Mj/Bkb6Y+hg30EQgghhOhHDg0lLAjsjm4t79N1HesfXsG69h0AzL+6Do/rFnY4Xg0LwnPVLZiuugDL/72M45ut2F75D7b3vkCNjUBLywRACQvCfMdVGC86HaWdJcyGmZNQYiPR84qw//cbTIvO7HLsR9MKS7F/tQU1PNhZijw28pgFM/T6Rhpve8KZ9E0bh3nlsh7H0BPGeTPxfu13NP7iMfTsAuov+xVez9yH8aTDyZLucND062egpg510hg8brvCfQGLIUGSqT4W4e9FiK+Z8joL+4urmRQT7O6QhBBCCNFM8fbEMGE03anHpes6lifXYvvnfwAw338jHldd0KlzDWNG4v3ib7F/uw3L/61D25+DllYHXmY8bliER+pCFG/PDs9XVBXT0rOdidxbH/c4mdJ1ncbbV7uSOdfjRIY4+zvFRaHERrr+rMZG0vSbv6Bl5KKEBeH1zH0oHqYexdAbDEnxeL/5expvewJtxz4alz+M+Tcr8LjEOZNnXfsujp92g5cnXk/d061qg0IcSf4G9TFFUUiODmTT/mLSDlW1SqYcmiYzVEIIIcQA0OUqbpqG5bEXncUhcO4V8rjs3C4/rvGUaRhOmox945doWYcwXXU+anhIp841LTwD65/Xo+3Yh2NfNoak+C4/fgv7Z987EykvM2pCDFpOIdQ1oBeV4ygqx/Hj7g6eQHPBibCB82WxGhqI98uP0fTgs9jf/wrLw8+hZeZhOnc21mede7s8f7McdaT0YRI9J8lUP0iODmLT/mLXvqkmm50fD5ZR02jlvMmxqKo0GRVCCCH6m1ZRjeLr3eUZFV3TsDz83OECBo/cimlx92eGFIMB08VndPk8NTQQ47yZ2D/+DttbH2P4Tecb5B5J1zSsf2ne73X1hZjvvNrZ66mq1tnrKedwnyctt7BVryfzb5ZjmHbsqoTuoJg98PzdXVgTYrD+eT22f/4H2+sfgt2B8dzZGBfOc3eIYoiQZKofJEcHAnCwtJZ6iw0vDyPVjVasdgeVDRZCfDuexhdCCCFE79MdDmfzW13HcEIyio93p89revBZ7O99DqqK5xO3Y1pweh9H2zHTJfOdydR/vsR8z7XdajRs/9+3aAdywM8Hj+suBpwzdUqQPwT5t1udT6+tR69vRI0M7elT6DOKomC+6RLUhBiafv1HaLKiRIXh+dub+7WxrhjaJJnqB6F+nkQFelFY1ciu/EpOSgxnxqhQ/DxN+Jjdv75YCCGEGHaarODt6Sw80dlEqqaOpoeexf7xZjCoeP7uLkznzenjQI/NcNJklJgI9Pxi7P/7FlMXZ1x0hwPrX98AwOPaBSgBvp06T/HzQfHz6XK87mA6exZqTCS2Nz/CdMX5KP6de45CdIZs2OknKSOd39xszS4DIDLAWxIpIYQQwk0UHy+MJ0zAMD25U+PtP+ykfuEdzkTKaMDz6V+5PZGC5kIUS5xNY61v/a/L59s/+BrtYD4E+OFxzYLeDm/AMCQn4Lnqlh7tKxOiPZJM9ZPpI52bSbfnVeDQpMeUEEIIMRAcr5qbbrHS9NQ6Gq9/CL2oDCU2Eu9XnsB09qx+ivD4TBefAUYD2ra9OA7kdPo83WbH0jIrdf1CFN/OzdAJIQ6TZKqfjIkIwNfTSL3Fzv4i56bNynoL23LKySiucXN0QgghxPChNzR1qhS6Y182DZf+EtvL7zmb0i6dj887z2CYOq7vg+wCNSwI4+knAmB76+NOn2fb+AV6XhFKcAAeV5zfV+EJMaRJMtVPVFVhWpxzdurnHOdSv+pGKwdLa8gur3VnaEIIIcSwoes6ju17cWzegV7X0P4YhwPr2ndouOQetP05KCEBeP31ATxX/QLFx6ufI+4c01LnUj/bxi/QmyzHHa9bbVifexMAj2WLB+zzEmKgk2SqHx25b0rXdaICvBkV6sfEEUHdahYohBBCiC5qtIBDc97aqXynHSqh8fqHsDz9D7DZMZ5+It7v/dk18zNQGWZNRYkOh5p67B9/d9zxtrc/QS8sRQkLwnTZOf0QoRBDkyRT/WhSTBBGVaG4pomCqgbMJgMp8aFEBnhLiU4hhBCiHyjenhhOmYphahKKweA6rus6to1fUH/xHTh+2g1enpgfuQXPZ+9HDQl0X8Cd5CxEcRYAtn8de6mf3mTBuuYtADyWL0Xx7Ho5dSGEkyRT/cjLw+jqObU1p9y9wQghhBDDlKKqrcp661U1NN39fzT9+hmoa0CdmoTPu8/gseTsQfVlp2nRGWBQcWxNx5GR2+E427/+h15SgRIZ6loeKIToHkmm+llKfPNSv+Z9UwANFjuZJTVYbA53hSWEEEIMebqj7fusrus0LHsY+/++BaMBj9uvxPuV1ahxUW6IsGfU8BCMc2cAYNvQ/uyU3tCE9W8bAPC4+VIUD2nTIkRPSDLVz1KaS6QfKKqhptEKwHcZxWzPLaeoptGdoQkhhBBDmmP7Puxb96DXHy48oReVoaVngtGA92u/w3zTJShGwzGuMrCZls4HwPbeF+gWa5v7ra9/iF5ejRIb2eUGv0KItiSZ6mchvp7Eh/qiA9tynUv9ooO8CfX1xGSQ/xxCCCFEX9CbLFBbDzV1YDo8G+NIywRATYzDMHGMu8LrNYZTpqJEhUFNXZtCFHp9A7a17wBgvvnS4/bYEkIcn3x6d4OWEunbmvdNjY8K5LRxUUQHSrM8IYQQoi8onmYMJ09BTU5otbRNS3cmU4YJie4KrVcpBsPhQhRH9Zyy/vN99KpalPhojBec5o7whBhyJJlyg+nN+6Z25lVgtTsG1eZWIYQQYrBSzB6o4SGtjrlmpoZIMgVgWnQmqCqOLWk4DuYDoNfUYX35PQDMt1w+qJcyCjGQSDLlBvGhvgR5e2Cxa6QXVLmOa5pObWPb9c1CCCGE6H26rqOlZwBgSB46yZQaEYLxtBOAw7NT1n/8G2rqUUfHYTx3tjvDE2JIkWTKDRRFcVX1a1nqV91o5f0duXy1rwhNGvgKIYQQvcax5yCOvVnoDU2tjuvF5ejl1WBQUZPi3RNcHzFd4ixEYf/352jF5Vhf+Q8AHrdejqLKxz8heov8a3KTlqp+P+eUoes6fubD67cbLHZ3hSWEEEIMKbrVhl5SgV5YCprW6j7XEr/RcUOuca1h9jSUyBD0qloaV6yC+kbUcaMwnnmSu0MTYkiRZMpNkqMDMRtVKuutZJfVoaoKp4+P5rwpsfh6Ss8HIYQYiA4cOMCsWbMYO3YsM2bMIC0trc0YTdO4++67SU5OZvLkyZx++ulkZGS4IVoBoHiYMExJQo2PRvFtXeipZYmfOoSW+LVQDAZMi50NebX9OQCYb7tCZqWE6GXyL8pNPIwGJsUEA4cb+Pp5mlClGIUQQgxYK1asYPny5ezfv5/77ruP1NTUNmM2btzIt99+y44dO9i5cydnnHEG999/f/8HK1yUQD/UUTFtjrfMTA2VSn5HMy06A5qTJ3XSGAzNDX2FEL3HrcnU7bffTnx8PIqisH379nbHfP7555x44okkJyczYcIE7r33XrTmafrs7GwMBgNTp0513TIzM/vxGfRMy1K/rc37po6ky74pIYQYUEpKStiyZQtXXXUVAIsXLyYvL6/NrJOiKFgsFpqamtB1nZqaGmJi2n6QF+6l6zqaK5ka7eZo+oYaFYbxvFPBaMB89zVSPViIPuDWbm1Llizh3nvvZfbsjqvKBAUF8cYbb5CQkEBTUxNnnnkmr7zyiuvbQD8/vw4TsYFu6sgQFCC7rI7yuiZCfD3Jr6hnf1E10UHejIsKdHeIQgghmuXl5REVFYXR6HzrVBSFuLg4cnNzGT368IfxCy+8kC+++ILIyEj8/PwYMWIEX331VYfXtVgsWCwW1881NTV99ySGGS2nEAAlKrRVbynAuY+qvGpIFp84kufjt6Pfdz1qSKC7QxFiSHLrzNScOXOO+23dtGnTSEhIAMDT05OpU6eSnZ3dD9H1vQAvD0ZH+AOHq/rZHBqVDRYKqxrcGZoQQohu2rJlC7t37+bQoUMUFBRwxhlncNNNN3U4fvXq1QQEBLhusbGx/Rjt0KVrGlpuIdrBPPS6tu+pWlrzfqnE2CFXfOJIiskoiZQQfWhQ7ZkqKipiw4YNXHDBBa5j9fX1zJgxg5SUFB555BEcDkeH51ssFmpqalrd3K2lge/PzclUVKA3KSNDOXl0uDvDEkIIcZTY2FgKCwux250VV3VdJzc3l7i4uFbjXnnlFebNm0dgYCCqqnLttdfyxRdfdHjdlStXUl1d7brl5eX16fMYTtTEWJTQIJQg/zb3uSr5DcHiE0KI/jNokqmamhouvPBC7r33Xk44wdmILioqikOHDvHTTz/x6aefsmnTJp5++ukOrzEQv/2b1rxvKv1QJU02O54mA6PC/PA0uXUFphBCiKOEh4eTkpLCq6++CsDbb79NTExMqyV+AAkJCXz++edYrc4m7O+//z4TJ07s8Lpmsxl/f/9WN9FziqqiRodhmDSm3b1CjvShvV9KCNE/BkUyVVtbyznnnMNFF13E3Xff7TpuNpsJD3fO4AQHB3P99dezadOmDq8zEL/9GxHoTYS/J3ZNZ1depbvDEUIIcQxr1qxhzZo1jB07lieffJJ169YBsGzZMjZu3AjALbfcwqhRo5gyZQqTJ0/ms88+4/nnn3dn2KId2hCv5CeE6B8Dfvqjrq6Oc845h3POOYff/OY3re4rKSkhKCgIk8mExWLhnXfeYdq0aR1ey2w2YzYPrHXRiqKQMjKUj3bl83NOGTMSwtA0nYKqBkpqGpkaF4KqSvUdIYQYCJKSkti8eXOb4y+99JLrz2azmb/97W/9GZY4ilZQCiaDc4lfO7NSWkk5elklqCpq0ig3RCiEGCrcOjO1YsUKYmJiyM/PZ/78+a6lEkd+w/enP/2JH3/8kXfeecdV/vzxxx8H4JtvvmHatGlMmTKFlJQUIiMjeeCBB9z2fLorJd651G97bgWapoMC23LLySqrpaLecpyzhRBCCNFCdzjQMvPQdmegl1e3O6ZlVkpNiEHxGlhfsgohBhe3zkytWbOm3eNHfsP3wAMPdJggLVq0iEWLFvVJbP1pbEQA3h5GaptsHCipISkygFGhvmg6mE0Gd4cnhBBCDB6ajjIiHL2iGiUkoN0hruITssRPCNFDg2LP1FBnNKhMjQsGYGt2GQATY4KZHBuMn6fpWKcKIYQQ4giKyYghIQbjCRM6bFLrKj4hlfyEED0kydQAkTLSWSJ9a3OJdCGEEEL0DVePKankJ4ToIUmmBogpscEYVIWCqgaKqp3NBXVdp6bRSnWj1c3RCSGEEAOfVlDaboPeVmNKK9BLncUnDOOk+IQQomckmRogvM1GxkU513a3zE7tL6rmk7RD7C2scmNkQgghxMCnW6xo+7Nx/LQbvaGpw3Gu4hOjRqB4e/ZXeEKIIUqSqQHEtdSved9UiK+nlEUXQgghOsOhoYQGogT5HzNJkuITQojeJMnUAJIy0lkifV9RNXVNNoJ9zVw4JY6ZCeFujkwIIYQY2BRvTwwTx6BOSTrmOEe6c7+UQfZLCSF6gSRTA0i4vxcxwT5oOuzIq0BVFIwG+U8khBBCdFZHFfxauJb5SSU/IUQvkE/qA8z05tmpn5uX+rXQNN0d4QghhBADnlZUhm6zH39caSV6SQUoihSfEEL0CkmmBpiWfVM78iqwOzQ0TefbA8X8Z3suTTaHm6MTQgghBha9pg5tz0Ec3+9Edxz7fVJLP6L4hI9Xf4QnhBjiJJkaYBLC/Qj09qDJ5uCHg6WoqkKjzY5d0yitbXR3eEIIIcTAomng640SEohiMBxzqEP6SwkhepkkUwOMqiicNWEEAO/vyEXXdabEhnBm8ghignzcHJ0QQggxsCiB/hhnTERNGnncsS37pQxSyU8I0UskmRqAzkyOxmxUyS2vZ1d+JWF+ngR4exx3U60QQggxXB1vVgrAkS7FJ4QQvUuSqQHI19PE6eOjAefslBBCCCFa0zUNrbwKXe9cgSatrAq9uNxZfGJ8Qh9HJ4QYLiSZGqDOnRSDqkDaoSqySmupbrCyK7+CrNJad4cmhBBCuJ1eXoW2cz+OrXs6NV5r2S8lxSeEEL1IkqkBKtTPk5NHRwDO2amKegv7i6rJKa9zc2RCCCHEAGBzgNGIEuTfqeGyxE8I0RckmRrALpgSC+Cq6hcX4ktiuJ+boxJCCCHcT40OwzBrCmpsZKfGS/EJIURfkGRqAIsL8WVybDC6Dl/sKWDGqDBig33dHZYQQggxICgGA4rJ2KmxjnQpiy6E6H2STA1wLbNTX+0roqbR6uZohBBCCPfSbXb0+q71XdTKq9CLWopPjOqjyIQQw5EkUwNccnQgCWF+WO0an6QdoslmJ7e8rtPVi4QQQoihRC8sxfHjLhwHcjp9TssSPzU+GsXHu69CE0IMQ52bGxduoygKF0yJ5c+fpvPx7kOu436eJoJ8zG6MTAghhOh/usW5SqMrSZEUnxBC9BVJpgaBGaPCCPf3pKSmibLaJkZHBGDXNHeHJYQQQvQ7w5iR6DGR4NH5jzAtZdENsl9KCNHLZJnfIKCqCudPdu6dSi+oZu64SML8pEeGEEKI4UnxMqMYDJ0e72hZ5ieV/IQQvUySqUFiTlIk/p4myuqa+PFgqbvDEUIIIfqVrmnd2i+sVVSjF5UBYBif0NthCSGGOUmmBgkPo4GzJ44A4P0deWiahs0hS/2EEEIMD3p5FY5NW7tUeAIOF59Q4qNRfKX4hBCid0kyNYicOWEEZqNKdlkd677Zz468CneHJIQQw8qBAweYNWsWY8eOZcaMGaSlpbUZs27dOqZOneq6hYaGsmjRIjdEO7ToVXXgcHT5PIfslxJC9CFJpgYRP08Tc8dFAbA7v4ryuiY3RySEEMPLihUrWL58Ofv37+e+++4jNTW1zZjrrruO7du3u26RkZFceeWV/R/sEKOOjsVwwgTUERFdOk9rruRnkEp+Qog+IMnUIHPu5BhUBUpqmxgT7u/ucIQQYtgoKSlhy5YtXHXVVQAsXryYvLw8MjIyOjznhx9+oKSkhAULFnQ4xmKxUFNT0+om2lIUBcXPB8Xbs0vnSfEJIURfkmRqkAnz8+KkxHAAPtyV7+ZohBBi+MjLyyMqKgqj0VmSW1EU4uLiyM3N7fCctWvXcvXVV2MymTocs3r1agICAly32NjYXo99uNIqa9ALnUWbpPiEEKIvSDI1CJ0/xflG+0NmCSU1jZ06p7rByse7D/HhzrxuVUMSQgjRNfX19bzxxhvccMMNxxy3cuVKqqurXbe8vLx+inDw0ApKcRzMR69r6Np5zfullJHRKH4+fRGaEGKYk6a9g1B8qB+TYoLYlV/J2q/388tzJ2EytM2L6y02tmSV8V1mCWmHKmnJoWKCfJgcG9zPUQshxOAWGxtLYWEhdrsdo9GIruvk5uYSFxfX7vi33nqLCRMmkJycfMzrms1mzGZzX4Q8ZGhFZVBdi+5p7lJFvpYlfgZZ4ieE6COSTA1SF0yJY1d+JXsKqsgurWFMZCAAFpuDbbnlbM4oYXtuOXbt8CyUp8lAk83BgeIaSaaEEKKLwsPDSUlJ4dVXXyU1NZW3336bmJgYRo9uv0rc2rVrjzsrJTpHjQ5zJlJBXdsrLMUnhBB9TZKpQWrCiECiA70oqGrkx6xy6iwONmcU83NOOU22w6VjRwR5c/LocE5ODGd7bgX//C6Dg6WyuVkIIbpjzZo1pKam8sQTT+Dv78+6desAWLZsGQsWLHAVmti3bx/bt2/nww8/dGe4Q4YaGQqRoV0+T4pPCCH6miRTg5SiKCyaPopnP0vnw515fLjz8Br7UF8zJ4+OYNbocGKDfVAUBYDEMBsAB0tr0XXddVwIIUTnJCUlsXnz5jbHX3rppTbjamtr+yss0Q69qga9oASQmSkhRN9xawGK22+/nfj4eBRFYfv27R2OW7t2LWPGjCExMZEbb7wRm83WqfuGuhMTQokK9ALA38vE2RNG8PBF03jmipO4bGYCcSG+rRKmkaG+GFSFmkYb5XUWd4UthBBCdJpeVYtu7fp7e8uslBIXJcUnhBB9xq3J1JIlS/jmm28YOXJkh2OysrJ48MEH2bRpExkZGRQXF/Piiy8e977hwKCqPHjhNG4/M5kbTh3LNaeMZkxkQIczTh5GAzFBzjeUg6XyjakQQoiBTdd1HDv34/h2G3p9A3pVrbOqn8V63HOl+IQQoj+4NZmaM2cOMTExxxyzYcMGFixYQGRkJIqicNNNN/H6668f977hwmwyUFDdQE55HYXVxy+TnhDmB0gyJYQYPuLj43nkkUeO2Q9KDFBWG3h6gNGIrqrUL7iNhgtuoS7lEurm3UDDdb+h6bd/xbL2HWwff4djbxZ6vfO9UEt3lkWXJX5CiL404PdM5ebmtpq5io+Pd70hHuu+9lgsFiyWw8vbhkKXeU+TgbERATg0nVDf45fWTQj344u9hVKEQggxbNx55528/PLLPPLII5x++unccMMNXHzxxVKOfBBQzB4YT5yE7nDg2JKGXlbpvEPX0YvKcBSV4fhhV9vzQoPQa+sBUCe0X21RCCF6w7Bq2jtUu8xPGBHE5NhgPIyG445tmZnKKq1Fk+a9Qohh4M4772T79u38+OOPjB8/nttuu42oqChuvfVWtm7d6u7wRCcoBgOOn9MBMJ47G59N/8D7td/h+eSdeNx8KcYLTkOdPBYl0Pkep5dVgsUKnh4YkhPcGboQYogb8DNTcXFxZGZmun7Ozs52NUg81n3tWblyJXfffbfr55qamiGTUB3pWJX6YoJ8MBlUGqwOiqsbiQrsfPNDIYQYzFJSUkhJSeHpp5/mueee47777uP5559n0qRJ3H777Vx33XVS5XQAa0mmDCdMQA0JhJBADFPHtRmn19Sh5RWh5RaijoxG8fft50iFEMPJgJ+ZWrx4MRs3bqSoqAhd13nhhRe47LLLjntfe8xmM/7+/q1uQ0mD1c6WrFLSC6o6HGM0qIwMcb6xyL4pIcRwYrPZ+Ne//sWCBQu45557OOGEE3jppZdYvHgx999/P1deeaW7QxRH0BuasH+33bkPymbHsX0vAIbpycc8T/H3xTBhNKZzT5X9UkKIPufWZGrFihXExMSQn5/P/PnzXV3kly1bxsaNGwFISEhg1apVnHLKKYwePZqwsDBWrFhx3PuGo6p6CznldRworsFqd3Q4LiFcilAIIYaPrVu3tlraN2HCBHbv3s0333zDddddx4MPPsinn37Ku+++6+5QxRH0qlqwWNEbmtD2HIRGC/j7oo7ueAWKEEL0N0XXh+/GmZqaGgICAqiurh4Ss1S6rrMzr4KYYB9CfD07HLdpfxEvfLGXsZH+/PailH6MUAghnPrz96/BYOCss87ihhtuYOHChZhMpjZj6uvrufXWW1m3bl2fxtIZQ+29qbt0hwO9ug4UBft7n2P5v3UY5s7A+7nfuDs0IcQQ1Z3fvwN+z5ToPEVRmBIXctxxic1FKLLL6nBoGgZ1wK/2FEKIbjt48OAx+xkC+Pj4DIhEShymGAwowQHA4f1SxuMs8RNCiP4mn6KHMLtDo72Jx8hAbzxNBqx2jUOVDW6ITAgh+k9JSQk//PBDm+M//PADW7ZscUNEoit0TcOx9XDxCSGEGEgkmRqiskpr+WhXPkXtNPJVFYVR0rxXCDFM3HLLLeTl5bU5fujQIW655RY3RCSOR6+qRcsrQq9vQDuY79w/5emBOl7KnAshBhZJpoaoOosNq91Bdln7yVJLv6nMEmneK4QY2tLT00lJabs/dNq0aaSnp7shInE8Wkk5WkYuWkEZji1pABimJKF4tN3vJoQQ7iR7poaopMgAfM0mRoa2318j8YjmvUIIMZSZzWaKi4tJSGg9q1FYWIjRKG+DA5Hi5wMhgShB/of7S6XIfikhxMAjM1NDlIfRwKgwP9QOGlC2lEfPrajH5tD6MzQhhOhXZ599NitXrqS6utp1rKqqivvvv5+zzjrLjZGJjqhRYRgmj0UNDTycTEnxCSHEACRfyQ0Duq5T22TD38vDdSzU1xM/TxO1TTZyy+tIDB++5XeFEEPb73//e+bMmcPIkSOZNm0aANu3byciIoJ//vOfbo5OHIt2qAS9qAyMBgxTx7k7HCGEaENmpoY4q93BV/uK+HxPIQ1Wu+u4ckQRiswSWeonhBi6RowYwc6dO3nqqadITk5m+vTp/OlPf2LXrl3Exsa6OzxxFL3Rgq45V0y0zEqp4xNQvDvunyiEEO4iM1NDnMmg0rLQr6regrfH4f/kiWF+7Myr4GBpDTDCLfEJIUR/8PHxYfny5e4OQ3SCIz0T6hpQJ4zG8XNz8YnpUhJdCDEwSTI1xCmKQkp8KAZVaZVIweGKflIeXQgxHKSnp5Obm4vVam11fMGCBW6KSBxN13WwWEHTUHy8ZL+UEGLA61YylZeXh6IoxMTEAPDjjz/y2muvkZycLN/8DUB+nu2Xkm0pQlFQ2UCTzY6nSXJrIcTQc/DgQS6++GJ27dqFoiiuZuZKc4Eeh8PhzvDEERRFwThrKnpDE3pjE9rBfACM08e7OTIhhGhft/ZMXXHFFXzxxRcAFBUVcdZZZ/Hjjz/ywAMP8Mgjj/RqgKJ3Ndnsrr1Tgd5mgn3M6EBWaZ17AxNCiD5yxx13MGrUKEpKSvD29iYtLY2vv/6aE044gS+//NLd4Yl2KN6eOLY275caHYcSKEWShBADU7eSqd27d3PiiScC8K9//YuJEyfy3XffsX79el5++eXejE/0or2FVXywI5/9RYfLA8tSPyHEULd582YeeeQRQkNDUVUVVVWZPXs2q1ev5vbbb3d3eKIDssRPCDEYdCuZstlsmM1mAD799FPXevNx48ZRWFjYe9GJXhXo7QHoNB5R1e9wMlXjpqiEEKJvORwO/Pycv+tCQ0MpKCgAYOTIkezbt8+doYkj6LqOfeseHAdy0O0OSaaEEINCtzbJTJgwgRdeeIHzzz+fTz75hEcffRSAgoICQkJCejVA0XvC/bw4Z1IMPubDe6ha9k3JzJQQYqiaOHEiO3bsYNSoUcycOZOnnnoKDw8PXnzxRRISEtwdnmhRWw/Vtej1jehRYWjpBwFJpoQQA1u3ZqZ+97vfsWbNGubOncvll1/OlClTANi4caNr+Z8YeFRVaZVIweGZqZKaJmqbbO4ISwgh+tRvfvMbtOa+RY888ghZWVmceuqpfPjhh/z5z392c3TCxcsTdUIiakIM2o59zop+0eGoUWHujkwIITrUrZmpuXPnUlZWRk1NDUFBQa7jy5cvx9vbu9eCE31H03UUwMdsIsLfi+KaRrJKa5kcG+zu0IQQolfNnz/f9efRo0ezd+9eKioqCAoKclX0E+6nmIwo4c7VLba3PwFkVkoIMfB1a2aqsbERi8XiSqRycnJ45pln2LdvH+Hh4b0aoOh9u/Ir+GBHHmV1FkCW+gkhhi6bzYbRaGT37t2tjgcHB3crkTpw4ACzZs1i7NixzJgxg7S0tHbH7dq1i7lz5zJ+/HjGjx/PO++80634hyvXfqkTpFmvEGJg61YyddFFF/HKK68AUFVVxcyZM3n66adZuHAhzz//fK8GKHqfxebAandQWNUAQKIUoRBCDFEmk4m4uLhe6yW1YsUKli9fzv79+7nvvvtITU1tM6ahoYGLLrqIxx57jD179rB7925OPfXUXnn8oUpvsqAVlqI3WtCtNhw79wMyMyWEGPi6lUxt3brV9cawYcMGIiIiyMnJ4ZVXXpH154PA6Ah/Th0bycQY58ziqJZkqkRmpoQQQ88DDzzA/fffT0VFRY+uU1JSwpYtW7jqqqsAWLx4MXl5eWRkZLQa99prr3HSSScxe/ZsAAwGA2FhHe/7sVgs1NTUtLoNN3p5NdreLLS9B9HSMsBiRQkOQB01wt2hCSHEMXVrz1RDQ4OrzOzHH3/MokWLUFWVk046iZycnF4NUPS+QG9zq5/jQ/1QFKhssFJZbyHIx9zBmUIIMfg8++yzZGRkEB0dzciRI/Hx8Wl1/9atWzt1nby8PKKiojAanW+diqIQFxdHbm4uo0ePdo1LT0/HbDZzwQUXkJ+fz+TJk3n66ac7TKhWr17NqlWruvnshgiTAfx9UYICsH3yHQCGlPGyp00IMeB1K5kaPXo07733HhdffDH/+9//uOuuuwDnt3b+/tKlfLDxNBkYEeRDfkU9maW1nCDJlBBiCFm4cGG/Pp7dbufTTz/l+++/Jzo6mvvvv5+bb76ZDRs2tDt+5cqV3H333a6fa2pqiI2N7a9wBwQ1PAS1ufiE9JcSQgwm3UqmHnroIa644gruuusu5s2bx8knnww4Z6mmTZvWqwGKvmF3aGSV1lJS28TJo8NJDPMjv6KerNJaTogPdXd4QgjRa37729/2ynViY2MpLCzEbrdjNBrRdZ3c3Fzi4uJajYuLi+P0009nxAjnErWrrrqqVUXBo5nNZsxm+RILQHc4cGzbC0gyJYQYHLq1Z2rJkiXk5uayZcsW/ve//7mOn3HGGfzxj3/steBE31EVhT2FVRRVN1Be1+TqN3WwZPit1RdCiM4IDw8nJSWFV199FYC3336bmJiYVkv8AC655BJ++ukn196nDz/80NWPUbSl2x3oug6AdiDX2bzX2xN1nDRUFkIMfN2amQKIjIwkMjKS/Px8AGJiYqRh7yCiqgpJUYEYVQU/T49W5dF1XZd16kKIIUNV1WP+TutKpb81a9aQmprKE088gb+/P+vWrQNg2bJlLFiwgAULFhAXF8f999/PrFmzUFWVESNG8OKLL/b4eQxVWmYeekkFamLs4SV+08ahGA1ujkwIIY6vW8mUpmk89thjPP3009TV1QHg5+fHPffcwwMPPICqdmvCS/SzpMgA159jg30xqAp1FjultU2E+3u5MTIhhOg97777bqufbTYb27Zt4x//+EeXCz8kJSWxefPmNsdfeumlVj9fffXVXH311V0PdhjSa+vBbgeTEcfPzr5dhunSX0oIMTh0K5l64IEHWLt2LU8++SSnnHIKAN988w0PP/wwTU1NPP74470apOh7JoPKyBBfDpbWkllSK8mUEGLIuOiii9ocW7JkCRMmTODNN9/khhtucENUooUhZTzUNqB7e+LYIsUnhBCDS7eSqX/84x+89NJLLFiwwHVs8uTJjBgxgl/84heSTA0imqZTXNOIzaGREObHwdJaDpbWcPLocHeHJoQQfeqkk05i+fLl7g5j2FNUFQJ80XMK0csqwWjEMGmMu8MSQohO6VYyVVFRwbhx49ocHzduXI+bIor+VVLbyHcZxXiaDK7mvVml0rxXCDG0NTY28uc//9lVcU/0P72mDny9nckUHF7iN3kMiqdUNxRCDA7dSqamTJnCs88+y5///OdWx5999lkmT57cK4GJ/hHu54W/lwdhfp4EeHkAkFVWh6bpqKoUoRBCDH5BQUGtClDouk5tbS3e3t6uynyif+kWK47t+8BswjAlCcXTjL2l+ESKLPETQgwe3UqmnnrqKc4//3w+/fRTV4+pzZs3k5eXx4cfftirAYq+paoKZ01wfjOraTpmo0qTzUFBdQMxQT5ujk4IIXruj3/8Y6tkSlVVwsLCmDlzJkFBQW6MbBhrtIBBBYMBzM4v8qRZrxBiMOpWMnXaaaexf/9+/vrXv7J3r7O53qJFi1i+fDmPPfYYp556aq8GKfqHqirEh/qxr6iagyW1kkwJIYaE1NRUd4cgjqIE+mE4cRLYHSiKglZagZ5bCIqCYVrbbQRCCDFQdbvPVHR0dJtCEzt27GDt2rXST2OQqmuyERvs40ymSmuYkxTp7pCEEKLH1q1bh6+vL0uXLm11/K233qKhoYFrr73WTZENb4rJCCbnx5CWKn7q2HgUf193hiWEEF3i1oZQBw4cYNasWYwdO5YZM2aQlpbWZsy6deuYOnWq6xYaGsqiRYsAyM7OxmAwtLo/MzOzv5/GkJB2qJL/7c7H3Nwk8aAUoRBCDBGrV68mNDS0zfHw8HCeeOIJN0Q0fGnZBehVbd9fZImfEGKw6vbMVG9YsWIFy5cvJzU1lQ0bNpCamspPP/3Uasx1113Hdddd5/p54sSJXHnlla6f/fz82L59e3+FPGSF+JpRFIUwP2cFpdzyOuwODaNBGjALIQa33NxcRo0a1eb4yJEjyc3NdUNEw5NeVYuWlQ+AYeZkFG9P132uZOoEadYrhBhc3PZJuaSkhC1btnDVVVcBsHjxYvLy8sjIyOjwnB9++IGSkpJW/a1E7wj38+L8ybGcOWEE3h5GbA6dvIp6d4clhBA9Fh4ezs6dO9sc37FjByEhIW6IaJjy8UKJCnPejkik9Jo6tP3ZgMxMCSEGny7NTLUsr+tIVVVVp6+Vl5dHVFQURqMzBEVRiIuLIzc3l9GjR7d7ztq1a7n66qsxmUyuY/X19cyYMQOHw8HChQt54IEHMBgM7Z5vsViwWCyun2tqajod71Cnqgpm1fm6JYT5sftQJQdLa129p4QQYrC6/PLLuf322/Hz82POnDkAfPXVV9xxxx1cdtllbo5u+FBMRgzjRqHreqvjjm17QddR4qJQw6S6ohBicOlSMhUQEHDc+6+55poeBdSR+vp63njjDb7//nvXsaioKA4dOkR4eDgVFRVceumlPP3009x7773tXmP16tWsWrWqT+IbSuJDfZuTqRrOINrd4QghRI88+uijZGdnc8YZZ7i+wNM0jWuuuUb2TPUDXdNcjXmBVmXq4YhmvTIrJYQYhLqUTK1bt67XHjg2NpbCwkLsdjtGoxFd18nNzSUuLq7d8W+99RYTJkwgOfnwL1uz2Ux4eDgAwcHBXH/99bz22msdJlMrV67k7rvvdv1cU1NDbGxsrz2noWBbTjkV9c7Zu4MlUoRCCDH4eXh48Oabb/LYY4+xfft2vLy8mDRpEiNHjnR3aEOerus4tu9D8fZETYx1VvA7SkuzXqMkU0KIQchtBSjCw8NJSUnh1VdfJTU1lbfffpuYmJhjLvG74YYbWh0rKSkhKCgIk8mExWLhnXfeYdq0aR0+ptlsxmw29+rzGGocuo6/l3MZZX5lPRabA7Op/WWTQggxmIwZM4YxY8a4O4zhpboWqmvR6xogPtpVCr2FlleEtsu5V1qKTwghBiO3lmpbs2YNa9asYezYsTz55JOuma9ly5axceNG17h9+/axfft2Lr300lbnf/PNN0ybNo0pU6aQkpJCZGQkDzzwQL8+h6FmbIQ/502OJcDLA02Hh97byqvfZbAtp5xGq93d4YlBwO7QeOXbA2zOKHZ3KEIAzgJHv/vd79ocf+qpp9r0nhK9Swn0xzBtHGpSPIpn6y8ztewCGq65H+x21ORElFjpbSiEGHwU/eidoMNITU0NAQEBVFdX4+/v7+5wBpS3fsriva05rY4ZVIWEMD8mjAhiwohAxkQEYJLS6S61TTb+sz2XsyaMIMzP8/gnDFGbM0p49rN0vDwMvHDNKVJeX7SrP3//hoWF8fnnnzNp0qRWx3ft2sWZZ55JcfHASvyHw3uTIzOPxusfRC+tRE2Mxevvj0rxCSGE23Xn969b+0yJgWvpjFGcPXEE6YeqSDtUSVpBJSU1TRworuFAcQ3vbc3Bw6gyNjKAiSOCmBIbTFzI8O5a//72XD7YkUdJTSN3nj3R3eG4zdacMgAarQ7SC6qYHBvs5ojEcFdXV4eHh0eb4yaTSaq69hG9pg68PNvdI+U4kONMpMqrUceOxGvtI6ghgf0fpBBC9AJJpkQbNofGgeJqSmubOHVsJCePdhb5KK1tJK0luTpUSXWjjd35lezOr+SNHw7ymwunMj460L3Bu1F6QRUAO/MqsNodeBiH314zu0Nje26F6+efs8skmRJuN2nSJN58800eeuihVsffeOONVkWNRO/QHQ4cuw4AYJiShOLr7brPsecgjTc8hF5Vizo+Ae+1q1ACh+bsmxBieJBkSrRhUBQyS2qx2h2U1TYR7u8FQJifF3PHeTF3XBS6rnOosoHdhyr5cm8heRX1bM0pG7bJVKPVTnaZs/qhxa6RdqiKaSOHXzPQfUXVNByxt25rTjmps/U2pZCF6E8PPvggixYtIjMzk3nz5gHw2Wef8dprr7FhwwY3RzcENVqg5d/8Ec15HbsP0LDsYaipQ500Bu8XH0YJGN4rGoQQg58kU6INVVUYFxmA2WQgxLf96oeKohAT7ENMsA9+niae+3yPa2ZmODpQXIN2xO7DrTllwzKZ2pZTDsDJieFszSmjot5CdlmdNH8WbnXhhRfy3nvv8cQTT7Bhwwa8vLyYMmUKn3/+OcHBMnPa2xRfb4yzpqJbba7+Uo4d+2i48WGoa0CdmoT3mt+i+Pm4N1AhhOgFsjNctGtMZABxIb4Y1OP/FUluno3KKauj3mLr48gGpr2FVQCENiefW3PK0YZZbRdd1137pWYmhrmW9/2cXebOsIQA4Pzzz+fbb7+lvr6egwcPcskll/DLX/6SKVOmuDu0IUvxcLbZsP+cTsOy30JdA4aUZLz/9rAkUkKIIUOSKdFjQT5mogO90YE9w3R2ak9zMnXh1Dg8TQaqGqzDrunxoaoGimuaMBkUJsUEMT0+FJBkSgwcX3/9Nddeey3R0dE8/fTTzJs3j++//97dYQ0pRxcItv+4i8blq6C+EcPMSXi9+FsUH+8OzhZCiMFHkinRIU3Xya+oZ3NGMQ5NO+bYltmp4bjUz2p3kNmcOE2McVY2hOGXRGxtfr7J0UF4moxMjQtBVSC3op6SmsZeeYx9hVXkV9b3yrXE8FBUVMSTTz7JmDFjWLp0Kf7+/lgsFt577z2efPJJZsyY4e4QhxQt6xD2H3aiFZdj/247jTc9Ao1NGE6ZitdzD6J4D9+2EUKIoUmSKdExHXbkVVBQ1cChyoZjDh3OydSB4hocmk6QtwcR/l6HZ2Ryhlky1bxfKiXeuVfMz9NEUmRAq/t6Iq+ijkf/s51V722ltml4LicVXXPhhReSlJTEzp07eeaZZygoKOAvf/mLu8Ma0vTyKmhowv7DThp/8Rg0WTHMmY7Xsw+geLW/B1cIIQYzSaZEh1RVISkygHFRgYQepwltSxW/vIp6ahqt/RDdwLG3sBqAcVGBKIrC1LhgDKrCocoGiqqPnYQOFdUNVjKKnf16psUdLrzRm0v9PksvQNehwergP9tye3w9MfR99NFH3HDDDaxatYrzzz8fg2H4tSvob4ap4yA4AMtDfwWrDeO8mXj9eSWKuW2fLyGEGAokmRLHNDrCnwkjgvD2OHbhR38vD2KDnRuKh9u+qZbiE+OinbMwPmYT43pxRmYw2JZbjg6MCvUlxPdw4t2STO0trKKuB7NJTTYH3xwodv38cdohyuuaun09MTx888031NbWMn36dGbOnMmzzz5LWVnPEvsDBw4wa9Ysxo4dy4wZM0hLS2sz5ssvv8TLy4upU6e6bo2NvbPUdaBTTEbs//4CrDYM05Px/OO9rkIUQggxFEkyJXrNcFzqZ3doHGiekRkXFeg6njLMii+0VPFred4twv29iAn2QdNhe273E8vvM0totDoI9/ckKTIAm0Pj3Z9zehSzGPpOOukk/va3v1FYWMiKFSt44403iI6ORtM0PvnkE2pru14kZsWKFSxfvpz9+/dz3333kZqa2u64pKQktm/f7rp5eXn18NkMDlppBbZ3PgXA4/YrUUzSgUUIMbRJMiU6parBwtacMqx2R4djkkcEAZBWUNlfYbldZmktNoeGn6eJEYGHK1S1zMjsK6qmdogve7TaHezKd/43T2mnt9b05mM/92CW7vM9BQDMGx/NpTMTAPhqXyGFVcNjGaXoGR8fH66//nq++eYbdu3axT333MOTTz5JeHg4CxYs6PR1SkpK2LJlC1dddRUAixcvJi8vj4yMjL4KfdDQbXYcaRlY//oGWG2o08ZhOGGCu8MSQog+J8mUOC5d19mSXUZWaS05ZXUdjhsfFYACFFY1Ullv6b8A3ci1xC8qAEVRXMfD/DyJC/FB12FbboWbousfuw9VYrVrhPiaGRni2+b+lsRyZ14FNsexq0K2J7uslsySWgyqwpykSJIiA5gWF4Kmw1s/ZfU4/u44uvyzGDySkpJ46qmnyM/P5/XXX+/SuXl5eURFRWE0OmdbFEUhLi6O3Ny2e/gyMzNJSUlhxowZPPfcc8e8rsVioaamptVtsNErq9GyDmH79xcAmJcvbfU7UQghhipJpsRxKYrC6HB/YoJ8CDlGIQofs4mRoc4P08Nlqd/e5ud55BK/FtNHDo+lfttaqviNDGn3w9OoMD+CvD1osjlIO9T1WcvP9xQCcEJ8KAFezk3sl5w4CgX44WApWaX918+rot7CQ+/+zEPvbqXRau+3x+0vWaW1/G93PjvzKqistwzppNFgMLBw4UI2btzY69dOSUkhPz+frVu38u677/LCCy/wr3/9q8Pxq1evJiAgwHWLjY3t9Zj6muLrg2P7XrBYUZNGYZgz3d0hCSFEv5BkSnRKfKgfMxPDCfY5dmnbCdHOpX7pw2Cpn0PT2O/aLxXQ5v6WGZld+RXHXB45mGm6frgk+sjQdseoitLtPWRNNjvfNheeOCM52nU8LsSXWaPDAfjXjwe7HHd3lNY28ejGbWSW1HKwtJa3t2T3y+P2h3qLjbVf7+PBd37mlW8z+N2HO7n11c3c/Mp3PPGf7fzzuwy+2ltIVmntkP273BmxsbEUFhZitzsTaV3Xyc3NJS4urtU4f39/AgKcvxNiYmK4/PLL2bRpU4fXXblyJdXV1a5bXl5e3z2JvqJr2N7/GgCP5UtkVkoIMWzIzlDRq5JHBPLBzrxhMTOVXVZHk82Bt4eRuOC2y9viQ30J9jFTUW8h7VAV09rZTzTYZZXWUtVgxdNkcJXHb8/0kSF8ll7A1pxyrtN11E5+0NqcUUKTzUGEv1eb6y+ZMYrvD5ayM7+S9IJKkpsT+b5QXN3I4+9vp7zOQoCXiepGG//dnc/ssRHEh/r12eP2NV3X+eZAMa9tzqSmudpicnQgVQ1WCqsbqG2ykVZQRdoR/54VBaICvIkN9iEuxJcTRoUSE+TjpmfQv8LDw0lJSeHVV18lNTWVt99+m5iYGEaPHt1qXGFhIREREaiqSm1tLe+//z433HBDh9c1m82YzYO7B5Ptzf9BTR1KfDTGs092dzhCCNFvJJkSXWKxOcguryMmyBsfc9tyt0mRAagKlNQ0UVbbdNz+VINZS3+ppMgAVLVtcqAoCikjQ/g0vYCfs8uGZDLVMis1JTYYk6Hjie7kEUF4mgxUNVjJKq0lMdy/U9dvWeI3b3xUmwQs3N+L08dF8Wl6AW/+kMXDCwP75NvwgqoGnvjPdiobrEQFeLHygims35zJDwdLWbfpAL9dOK3TyeFAcqiynnXfHHC1MhgR5M11s8e6klar3cGhygZyy+vIragnr7yOnIo66prsFFQ1UFDVwA8HSwnz8xw2yRTAmjVrSE1N5YknnsDf359169YBsGzZMhYsWMCCBQt4++23ef755zEajdjtdpYuXcp1113n5sj7jqOwFOu69wAwL1uMIv28hBDDiCRTokt+zimjsKoBi83B5NjgNvd7eRhJCPMno6SG9IIq5iRFuiHK/tFSfGJ8dNslfi2mx4fyaXoB23LL0bowIzNYbG1etne8RNFkUJkSG8wPB0v5ObusU8lUVqlzOZ2xufBEexamjGTT/iIySmrYmlPuWlrZW/Iq6lj9/g6qG22MCPLm/gumEOht5qpZo9mRV0FGSQ1f7ilk3hFLEAc6i83Bv7fl8P6OPByajodR5eKUkZw3ORbjEQmxh9HAqDA/RoUdnnnTdZ2qBit5FfWuJCshbPDOzHVHUlISmzdvbnP8pZdecv351ltv5dZbb+3PsNzK9vK/0curUMKDMV5wmrvDEUKIfiV7pkSXjAr1I9DbTKB3x93sD/ebGrr7pjRdZ1+Rc2aqveITLZKjA/HycM7IZJYMvgpdx1Ja20huRT2KAlPjjj/rNr2L+6ZayqHPGBWGv1f7f9+CfMzMnxgDOPdOaVrvFUzILqvlsf9sp7rRRlyID7+5cCqB3s6lWME+ZpbOGAXAGz8epHqQlL/fnlvOfW/9xL+35eLQdKbFhfDUJTNYMG1kq0SqI4qiEORjZnJsMBdMjeMX88YTdURLADH8aE0W7P/9BgDTDYukQa8QYtiRZEp0SWSAF/PGRxHXTgnsFskjAgFnRb+hWg0sv6Keeosds1ElPrTj18LYPCMD8HN29/ssDUQtS/ySIgPw8zz+B6ipccGoCuRXNlBc3XjMsY1WO99llAAwLznqmGMvmBqLj9lIfmUD32YUdzL6Y8ssqeHx/+ygrslOQpgfD1wwtU1Cd9aEaEaG+FJvsfP695m98rh9pbyuiWc+3s3/fbSL0tomQnzN3HX2BO45ZyJhfsOjmazoG47/fYteWokSEoDH0rPdHY4QQvQ7SaZElyiKctx9KWMjAjCoCuV1Fkpqmvopsv61p3mJ39jIAAzqsf8ZdXVGZrDYmn3sKn5H8zGbGN88i/dzzrFfi++aC09EBXq5zjnWdS+c6qym9vaW7G71sjrSvqJqnnh/Bw1WO2Mi/Fl5/hR820kWDarK9aeORQE27S927T0aSOotNt7bmsO9//qJn7LKUBU4f3IsT10ygxNGhUnFNdEjuqZh/dvbAJiuWYDiObiLaAghRHdIMiW6RdN1CirrqW5ou7zJbDIwunlPTNoQXerXUnzieB/0wVmcwaAqFFQ1UFTd0MeR9Y8Gi92VUE6P73xhjc6USNd13bXEb9746E594D97wggCvT0orW1yndsd6QWV/O6DHTTZHIyPDuTX50/G29zx1tLREf6cPt45c/b3Tfux9zCR6y2V9RZe+z6T29d/z1s/ZdFkczAmwp/HF5/AFScn4mmS7bKi5+wff4d2MB/8fPC4/Dx3hyOEEG4hyZTolt35lWzOLGF/876hox3eN1XVf0H1E13XXcUn2usvdTQfs8k1bqgs9duRV4FD04kO9CYyoPN7Zlpm6fYVVVPbwT6jg6W1ZJfVYTIonDq2cwVMzCYDF08fCcB7W3NosnW9oe7OvAqe+nAXFrvGxJggfnXOpE4lHZfNTMDf00RBVQMf7nRvf6DCqgb+9tU+7nztez7YkUeTzUFMsA+/mDeehy6adszluUJ0ha7rWJ93NiI2nnMKiq/snRNCDE+STIluiQ32wcNo6PBb+5Z9U3uG4L6pwqoGahptmAwqCZ0s8T3UlvptbV6ml9LFcu9hfp7Ehfig67Att6LdMS3l0E8cFdapvVgt5iZFEeHvSU2jjY925Xf6PE3X+eFgCU//dxc2h8bUuGDumT8Rs6lz5Z19zCauODkRgHe35lBSc+z9YH3hYGkNf/okjV+9+SNf7i3ErukkRQbwy3Mm8eSSEzhlTMSQqyQp3Mvx7Ta0AzngYcK08Ax3hyOEEG4jaz1EtwT5mDlvckyH+4VGh/tjMqhUNVgpqGpgxBDqQ7OneYnf6Aj/Y/ZWOlLKyFBe+TaD/cXOGRm/DqrTDQZ2h8b25kQopRulyKePDCW3vJ6fs8valDxvsNrZ3FxEoqvlxo0GlSUzRvHXz/bwwY48zkwe0WEypus6WWW1fJ9RyvcHSyivswAwY1Qot56R3KnKdkeaPSaCr/YVsaegile+zeCecyb2+X4kXddJO1TJxu25pB2qch1PGRnCBVPjSIo8/qypEN1lXfMWAKYlZ2NITnBzNEII4T6STIluO1bhBQ+jgbER/qQVVJFeUDWkkilXf6lOLPFrEebnycgQX3LK69iaW85pSceuUDeQ7S+upsFqx8/TxJhOzswdaXp8KO9uzWFXfgVWuwMP4+EZoO8OFGOxa0QHencrGTgpMZz/bM8lt7ye/2zLdc0YgTP5yKuo5/vMEr7PLKH4iOIoniYDc8dFccVJCcctKNIeRVG4bvYYVm7Ywrbccn7OLuOEUWFdvk5n1DbZ2Jpdxsdph8guqwPAoCrMGh3OBVPiiAkeOv/WxMBk35KG4+d0MBnxuHExinnwfjkkhBA9JcmU6LGaRisOTSfIp3Ulp/EjAl3J1FkTRrgput6l67qr8MKx+ku1Z3p8iDOZyh7cyVRLFb9pcSGoatdnX+JDfQnxNVNeZ2H3oUpXNUBd1/msuXjEGcmdKzxxNFVRuPTEBP7vo118nHaI+ZNGYLFrfJ9ZwuaMEgqqDhcA8TCqTIsL4eTR4UyJDW6V1HXHiCAfzp8cy8btufzj2wwmxgT1WqGH8romfs4u46esMvYWVtHSTstsVJk7LopzJ8cS5ufZK48lxPG4KvgtPAM1omtLfYUQYqiRZEr0SFZpLVtzygjz82qzZGtCdBAbyCa9oBJN14fEno3S2iYq660YVMVVsbCzUkaG8s7P7c/IDBa6rh/eL9WFKn5HUhSFlJEhfJJWwM/Z5a5kKrOkltzyekwGlVPHRnQ7ximxwSRFBrCvqJr7N2yhznK4GIVRVZgaF8JJiWFMGxnS61XtFqaMZHNmCaW1TbyzJafVzFhXFVQ18FNWKVuyyjhYWtvqvrgQH2YmhHPG+KhBvWRUDD6O9Ewcm34GRcEw70R0q00a9QohhjVJpkSPRPh7oSoKJoOKpumtZioSwvwwG1XqmuzkV9QPiUpiLb2EEsL8Ol2goEVHMzKDyaGqBoprmjAZFCbFBHX7OtPjQ/kkrYBtOWVo+lhURXGVND8pMQwfc/c/nCmKwqUzE3jk39uos9gxqAoTRwRx8uhwpo8MPWap854ymwxce8oYfv/fXXy0K4/ZYyM6/fde13Wyy+r4KauUn7LKWs2iKcCYSH9OiA9jxqhQwv2l0a5wD+uLGwAwTE9GAeji/kIhhBhqJJkSPeJtNnLe5Nh2EwujQSUpMoCd+ZWkF1QNiWSqpb9UV5f4QcczMoPJ1uZqhMnRPVvCNj4qEC8PA9WNNjKLa4gO8mZzZgkAp4/vWuGJ9iRFBnDnWROot9qZHh/apaqAPTVtZAgnxIeyJbuMdd/s58EF09rMyjZanV8w5FXUk1tR5/z/8noarIdn0QyqwoToQGaMCiMlPoRAb2mIKtzLcTAf+yebAfC49QrU6DAUw+CbYRdCiN4kyZTosWPN0CSPCGJnfiVphyo5Z1JMP0bVsR8PlvLGD5lcPWsM07pY2ntPN4pPHKm9GZnBZGuOc79Ud5f4tTAaVKbGhrA5s4Sfc8rIKjNjtWvEBHkzNqLrRS3aMyOhbwpAdMY1p4xmV34F+4tq+GBHHuF+nuRW1JNXUUdueT2ltU3tnmc2qkyODWbGqDCmxgX3aIZOiN5mfelt0HWM82ZiPHGiu8MRQogBQZIp0WvsDg1N11vtBWpp3ru3sKrNMkB3KK9r4sWv9tJodfDXz9N5bNH0TjedLa9rorS2CUWBMd0sO330jEx3r+MO1Q1WMoprAGfxiZ6aHt+cTGWXY2j+ezGvm4UnBpoQX08WnzCK177P5I0fDrY7Jsjbg9hgH2JDfIkN9iEuxJfoQO9Ol9sXoj9pJeXY3/8KAI/lS9wcjRBCDBySTIlesb+omvSCKsZE+DNhxOG9NPGhvnh5GGiwOsgpr2NUmJ/bYtR1nb99tY9GqwOARquDP3+SzsMLp3WqGETLEr9RoX54e3Tvn87RMzLHS6ZqG62kF1aRV17PlNhgtyZf23LL0YFRob6E+Pa8ctyU2BAMquLaG+RhVJk9pvuFJwaa+RNHsC23nIMlNcQE+xAb3JI0Of/cn0sPhegp+5dbwO5ATU5ECQlAt1ilJLoQQuDmZOrAgQNce+21lJWVERAQwMsvv8yECRNajfnyyy8599xzSUpKch3bvHkzXl7ODdhr167lySefRNM05s2bx3PPPYfJJB9S+punyYBD06iot7Q6blBVxkcFsjWnnPSCSrcmU1/sLWRXfiUmg8ovz5nEs5+lk1Nexz+/y+CGOUnHPb+lv1RSN5f4tUg5Ykbmspmtq73VW2zsaS4nn15QRV5Fveu+97bmMH/iCJaemIBnF4tf9IbDVfx6Z6+Xt9nI+OhAdudXAnBSQviQWtZmNKj85sKp6Lo+JGbbxPDm+PpnAAwTx6Dtz0G12lBGDYyl20II4U5uTaZWrFjB8uXLSU1NZcOGDaSmpvLTTz+1GZeUlMT27dvbHM/KyuLBBx9k69atREREcNFFF/Hiiy9yyy239EP04kjRgd7MGx9NoHfbbyqTo53JVNqhKs6fEueG6JwlzddvzgTgkhNHMTEmiF/MG89TH+7k8z2FJEUGMHts5DGvcXi/VGCPYpkSG+yakckqraWqwUp6gbNIR05ZHfpR42OCfQj1NbM9t4L/7j7E1txybjwtieTo7lfTA2iyOfg5u4w6iw29+UH15j/o0HxMR9edP7ckPSld3Gd2LNNHhrquOy958PbeOhZJpMRgp1tt2L/fAYBhznQI8EMJ7dnvHyGEGCrclkyVlJSwZcsWPv74YwAWL17MrbfeSkZGBqNHj+7UNTZs2MCCBQuIjHR+CL7pppt44oknJJlyA6NBbdO0t8X45n1T+4qqsTs0jJ3cE6LrOg1We49nK5zL+/bSZHMwNtKfcyY6v02dHBvMwpSRvLs1h79v2k98mB8xQT7tXqO6wUphVSMKzkpxPeFjNjE+KpDdhyr5zTs/t7k/OtCb5OhAkqMDGRcdSEBzH6GdeRW89PU+SmqaePw/OzgzOZrLT0roclW92iYbH+8+xMdp+dQ12Y9/whFCfM2M7MWqjCcmhPLOz9nEhfh0uW+XEKJ/OH5Oh4YmlJBAjHNPQFFlX58QQrRwWzKVl5dHVFQURqMzBEVRiIuLIzc3t00ylZmZSUpKCgaDgeuuu45f/OIXAOTm5jJy5EjXuPj4eHJzczt8TIvFgsVyeBlaTU1Nbz4l0axldqPlG/m4EF98zUbqLHayymoZE3H8ZKSy3sJfP9/D3oIqFs+I5+KU+G7H89meAtIOVeFhVFkxd1yrIhiLpsezv7iatENV/OnjNB5dNL3dJXQtS/xig33w7YW9LrNGh7P7kHNGJsLfk/HRQa4EqqOkdHJsME8uncHr32fy+Z5CPk0vYHvzLNXEmODjPmZpbRMf7czjy72FWOwaAGF+niSE+aEooKDQ/D9n/xhFcf65+T5VhVPGRPTqTEugt5m/XHVy8+PIDI4QA5F901YADKemSCIlhBBHGfAFKFJSUsjPzycgIID8/HzOO+88QkNDueSSS7p8rdWrV7Nq1ao+iFK02FNQRW5FHSclhBPQvORPVRTGRwfyU1YZ6YeqjptM7cyr4PnP91DTZANgw0/Z2OwaS2eM6vIH7pKaRl5rXt536YkJbSr3qarCLfOSuf/tLRRUNfD3Tfu5+fRxbR6nJ/2l2jMnKZKoQG9CfM1dKubg7WHkhjlJzEwI56Wv91Fa28TqD3Zy+rgorjgpsd2GtLnldby/PZfNmSVozcv54kN9uWBKHCcmhGJw84cjqV4nxMDm2NS8X2p6MrqmSUIlhBBHcNtvxNjYWAoLC7HbncuMdF0nNzeXuLjWe2r8/f0JCHB++I6JieHyyy9n06ZNAMTFxZGTk+Mam52d3eb8I61cuZLq6mrXLS8vr7ef1rBX1WChrslGfmV9q+MtJdLTC6o6PNfu0Hjjh4P87sOd1DTZiAvxYcFU53/Pf2/LZf3mTNesV2dous6LX+3DYtcYFxXA2RNHtDsuwNuDW89IRlXg2wPFfLG3sM2Ylv1S46J7p5qeoiiMjQzodlW8iTFBPLn0BM6e4HxOX+wt5L63fmJHrrMPlK7r7Cmo4qkPd7Jywxa+zXAmUhNGBPLr8yfz2KLpnDw63O2JlBBiYNMOlaBl5oGqoHh64Ph+Z5d+DwshxFDntpmp8PBwUlJSePXVV0lNTeXtt98mJiamzRK/wsJCIiIiUFWV2tpa3n//fW644QbAuc9q9uzZPPzww0RERPDCCy9w2WWXdfiYZrMZs7n9JVSid4yOCGBEkA9Rga1ngJKby6XvL67G5tDazEaU1zXxl0/TOdDcx+jM5GiuPDkRD6OBIB8z//j2AB/tysfq0EidPaZTzW4/TTvEnoIqzEaV5aeNO+Y546MDueTEBN744SCvfHuAhDA/4kOdlQfrmmyuqnrjIgM7/Vr0NU+TkWtnj+HEhDD+9tVeimuaeOqjXZw8OpySmkYyS2oB5zK9E0eFccHUWBLCZF+SEKLz7M2zUuqE0SgBfij+PrIkVwghjuDWZX5r1qwhNTWVJ554An9/f9atWwfAsmXLWLBgAQsWLODtt9/m+eefx2g0YrfbWbp0Kddddx0ACQkJrFq1ilNOOQWAuXPnsmLFCrc9H+Hcg9OeEYHeBHiZqG60kVFc4ypKAbA1u4w1X+6lzmLHy8PAjXOSmJkY7rr/7Ikj8DCqvPTVPj5LL8Bm17jxtKRjNgAuqm5wNUu9bGYiEQFex439/Cmx7CusZltuOX/+JI3HFp2At9nI3iLnEr/oQG/X0sWBZHx0IKuXzOCtn7L47658NmeUAGAyKMxJiuK8yTGdbkwshBBHakmmjPNmYjhlGti6VrRGCCGGOkUfxvP1NTU1BAQEUF1djb+/fGPf1/7yaRrfZ5ayaHo8i0+Idy3r+2hXPgAJYX7cdmYy4f7tJz7f7C/ihS/3outw8uhwbpo7rt3KgJqu89jG7ewrqiY5OpCVF0zp1EwWOGehHnh7C2V1FmaMCuWOsyaw/vtMPtqZz7zxUZ3qR+VO+4uqeXdrDqNCfZk/MWZAJn9CgPz+PZaB8troVht1J18JjRa8N/wRQ3KC22IRQoj+0J3fvwO+AIUYfOwOjUNVDZTXNTEtLsS1JGRCdBDfZ5ayp6CSkpoI/vJpOgdLnUvRzpkUw2UzE45ZjGD22EhMRpW/fraHzRkl2Owat52Z3Cah+t+ufPYVVeNpMrB8blKnEykAX08Tt581gVX/3sZPWWX8b/ch9jbv8+qt4hN9aWxkAPedN9ndYQghhgDHljRotKCEBaGOH+XucIQQYkCS3eei1+nA1pwyskprqWm0uY4njwgEYH9xDQ+8vYWDpbX4mI3cPX8iV88a3amqbjMTwrnzrAkYVYUt2WX88ePdWO0O1/2FVQ28+WMWAFeclEiY3/GX9x0tMdyfK09OBOC17zPJLq8DYFxU7xSfEEKIwcC1X2psPI6te9AbLcc5Qwghhh9JpkSvMxlUEkL9GBcViMl4+K9YhL8XQT4eODSdBquDMRH+PL74BKbHh3bp+inxodxzziQ8jCrbcyt4+r+7abI50DSdNV/uxebQmDgiiHnjo7r9HM6eMIKZCWE4NB1dh3B/z25X3hNCDB0HDhxg1qxZjB07lhkzZpCWltbhWF3XmTdvHoGBgf0XYC9ytPSXGjcKGpqghw3UhRBiKJJkSvSJKXEhTBgRhLfH4ZWkiqJwcmI4CnDBlFh+c+HUDgtWHM/k2GDuPXcyZqPK7kOVPPXhTt7dmsOB4ho8TQZuPC2pRxWnFEVh2WlJRDYXrhhIVfyEEO6zYsUKli9fzv79+7nvvvtITU3tcOwf//hHEhMT+y+4XqTlF6MdzAeDiunai1AnJEp/KSGEaIf8ZhT96oqTEnkxdTaXn5TYbvGIrhgfHcivz5+Cl4eBfUXVvPNzNgBXzxpNaDeTtCN5exj51bmTOCM5moUpI3t8PSHE4FZSUsKWLVu46qqrAGd7jry8PDIyMtqMTUtL47333uPXv/71ca9rsVioqalpdXO3liV+hqnjUEMDUYNlmbMQQrRHkinRZ3Rdp6LeQlF1g+uYoih4m3uv7snYyADuv2AKPs3XnBwbzGlJkb12/cgAb64/dWynSqsLIYa2vLw8oqKi+P/27jw6yvLeA/j3fWeSmUz2fSEbIQtrFraWxbJ5BRRETKm0WIkbIOq9t94eLAcB6S0FW4vl1qNitdxjodAjYC8tatk8ohQhgAFkiUkgJGQhCVkmk2XW5/4RGRiykAyTvJnJ93POnDN532dmfs+8mfeZ3zzLq1a3nW8kSUJ8fDxKSkocypnNZjz77LPYsmULVCrVXZ93w4YNCAwMtN/i4uJ6Jf6esBz5Lpn6wRiFIyEi6t+YTFGvKatrxmcXy3G2tBa9uQJ/UngAXn1kNH40bjCWTx/GC0oSkaLWrVuHRx99FMOGDetW+ZUrV6KhocF+Ky0t7eUIuyaMJliPnwUAyIMiIQzNd3kEEdHAxaXRqddEBvrAW61CoK5t0Qm1qveSnJggHeZxKB4R9aK4uDhUVFTAYrFArVZDCIGSkhLEx8c7lPv8889RUlKCN998ExaLBXq9HomJicjNzUV4eHi759VoNNBoNH1VjbuynjwPtJqAIH9A6w1htoA/URERdYzJFPUaL5WMh9LjIMtshonI/UVERGD06NHYtm0bcnJysHv3bsTGxiI5Odmh3BdffGG/X1xcjMzMTBQXF/dxtM67OcRPfd9oyPHRkIL8FY6IiKj/4jA/6lVMpIjIk2zZsgVbtmxBamoqNm7ciK1btwIAnnnmGezdu1fh6Fzj5uIT6hnfhyo5nkOniYi6wJ4p6hNGc9uFdTVed5+MTUTUX6WlpeHYsWPttr/33nsdlk9MTER9fX0vR+U6tpIKiOJyQK2CekKG0uEQEfV77JmiXnehvA77zpaisEr55X6JiKhzlu8u1CsPSwK6sRIhEdFAx2SKep2fxgtCCBhazUqHQkREXbg5xE8eEgdbWZXC0RAR9X8c5ke9LiZIh5kjY+Gn9VI6FCIi6oRoNcJ6/BwAQD0hA3JkiMIRERH1f0ymqNepVTL8VOwEJSLqz6y55wGjCVJUKNRzpnDhCSKibuA3XOpTQohevYAvERE5x3LkJABAfd8YJlJERN3EZIr6zNnSWnx89hr0nDtFRNTv3Fx8QjUpU9lAiIjcCJMp6jMGoxmtZgvK6pqUDoWIiG5jKy6HKKkAVDLgo1U6HCIit8E5U9Rn0qICMTjMHxEBbKiJiPoT+yp+KQmQB0UqHA0RkftgMkV9JtSPSRQRUX90c4if15wpkBNjFI6GiMh9cJgfKcJitXEhCiKifkC0GGE90bYkuuq+MZBkfjUgIuounjGpz1XUN2P/+TKU1TcrHQoR0YBnPXEOMJkhRYVBTo5TOhwiIrfCZIr6XG2TES0mCwqv69k7RUSkMMvnbUuiy0MHA1abwtEQEbkXzpmiPjc0OhBqWcKQiABey4SISEFCCFiOtC0+ocpIhaRWKRwREZF7YTJFfU4ly0iLDlI6DCKiAU9cLYcorwLUang9er/S4RARuR0O8yPFXW9ogc3G4X5ERH3N3is1djjk8BCFoyEicj9MpkhRp6/W4MuCShRcb1A6FCKiAefm9aXUPxircCRERO6JyRQpKtRPC0mSYGHPFBFRnxLNrbCe+AYAoBo7QuFoiIjcE+dMkaLiQ3wR4quBv9ZL6VCIiAYUy4lzgNkCKTgAEpdEJyJyCnumSFGSJDGRIiJSgPWbAgCAPGY4ZK1G4WiIiNwTkynqN1pMFpwqroHRbFU6FCIij2c7kw8AUE/IUDgSIiL3xWF+1G8cv1yNG4ZWAMCYxDCFoyEi8lzCZoP1zLcAAFXmUIWjISJyX+yZon5j5KBghPhqMCTCX+lQiIg8mvV8EWBoBnw0kFMSlA6HiMhtKZpMFRQUYOLEiUhNTcW4ceNw/vz5dmUOHz6M8ePHY/jw4RgxYgRWrFgBm80GACguLoZKpUJmZqb9VlRU1NfVIBcJ89di6tBoBOk4dp+IqDdZvzoDAJAHD4KkVikcDRGR+1I0mVq6dCmWLFmCb7/9Fi+//DJycnLalQkODsbOnTtx4cIFnDp1Cv/617/wwQcf2Pf7+/sjLy/PfhsyZEgf1oBcTZIk+33rd0kzERG5li2/GACgyuAQPyKie6FYMlVVVYWTJ0/i8ccfBwBkZ2ejtLQUhYWFDuWysrKQlJQEANBqtcjMzERxcXFfh0t9SAiBy9V6fHL2GhpbTEqHQ0TkcWwXLwMA1D8Yo3AkRETuTbFkqrS0FNHR0VCr29bAkCQJ8fHxKCkp6fQxlZWV2LVrF+bMmWPf1tTUhHHjxmH06NH45S9/Cau185XgjEYj9Hq9w436p4r6FhgtVhRWNQJoS7BqGltR3dgCm7h1gV99iwnX6ppQ12RUKlQiIrci6vWwXSkDAKgyUhWOhojIvbnNAhR6vR5z587FihUrMHbsWABAdHQ0ysrKkJubi4MHD+KLL77A7373u06fY8OGDQgMDLTf4uJ4kcL+SJIkZMaHID0uBBnxIQAAIYDP8ytwJL8SFuut4X9ldc04XlSFKzWNSoVLRANId+b6Hjt2zD6Pd8SIEVi6dCmMxv7zg4/lxDcAvpsvFRSgcDRERO5NsaXR4+LiUFFRAYvFArVaDSEESkpKEB8f365sY2MjZs2ahXnz5uGll16yb9doNIiIiAAAhISE4KmnnsJf/vIXrFixosPXXLlypcPj9Xo9E6p+ylfjhZTIQPvfktS2Tb5tThUA+HirEOanhZ+GF/4lot53c65vTk4Odu3ahZycHOTm5jqUycjIQG5uLry8vGCz2ZCdnY233noLP/vZzxSK2pHl85MAADmVq/gR9QUhBCwWS5ejp6hvqFQqqNVqhzn690qxZCoiIgKjR4/Gtm3bkJOTg927dyM2NhbJyckO5QwGA2bNmoVZs2bhlVdecdhXVVWF4OBgeHl5wWg0Ys+ePcjKyur0NTUaDTQarhTnjiRJwqxRse22J4b5IzHs1lLqFqsNlQ0tiA3x7cvwiGgAuDnXd//+/QDa5vq+8MILKCwsdGi7dDqd/b7JZEJLS0uXDbfRaHTouertIej2xSeyhvXq6xBR2zmgoqICzc3NSodC39HpdIiOjoa3t7dLnk/Ri/Zu2bIFOTk5+PWvf42AgABs3boVAPDMM8/g4YcfxsMPP4zNmzfjxIkTaGpqwp49ewAACxYswKpVq/Dll19izZo1UKlUsFgsmD59OlatWqVklUhBFqsNB86XodlkgdYrGmH+WqVDIiIP0tVc3zt/CCwuLsa8efNQVFSEhx56CMuXL+/0eTds2IB169b1auw3CYvVPl9K/n56n7wm0UBls9lw5coVqFQqxMTEwNvb26U9ItQzQgiYTCZUV1fjypUrSElJgSzf+4wnSYjbZvMPMHq9HoGBgWhoaEBAAMeNe4LTxTW4rm/BmMQwRAT4KB0OEXXCHc+/p06dwk9+8hPk5+fbt40fPx4bN27E9OnTO3yMwWDA448/joULF2LhwoUdlumoZyouLq5X3hvrxctozv4Z4KeD31fbIbngiwQRday1tRVXrlxBQkKCQ481Kau5uRlXr17F4MGDodU6/vDuTNvEsyh5lJGxwXhg5CAmUkTkcrfP9QXQ5Vzfm/z8/LBw4UJs37690zIajQYBAQEOt95izbsEAFClpzKRIuojruj9INdx9fHg0SWP4q1WQcWTFhH1gtvn+gLodK5vYWEhzGYzgLb5Eh999BHS05UfUidsNlgOfgUAkNO5JDoRkSvwWyd5rMqGZhRV8VpiROQ6W7ZswZYtW5CamoqNGzc6zPXdu3cvAODw4cPIyspCRkYGsrKyEBkZidWrVysZdht9E2wFbddy5OITRESuoegCFES9pbqxFUcLrkOWJUQH6qDT8F+diO5dWloajh071m77e++9Z7+/ZMkSLFmypC/D6habyQRRUwdIEtS8WC8RkUvwGyZ5pDA/DSICfBDo4w21iivnEBHZzhUCAOQhcZAC/BSOhojIMzCZIo8kSRImp0T2uyVIW80W1DaZoPNWIUjHa54RUd+xnflu8YnMNIUjISLyHJwzRR5LyURKCAFDqxnX6ppw+9UHvq3U41jhdRTXGBSLjYgGHmFohuWrswAAVeZQhaMhImG1QlitjttstrbtNluHZW//PtHTss749NNPMXnyZAQFBSE0NBRz5sxBUVGRff+1a9fw4x//GCEhIfD19cXYsWNx/Phx+/6///3vGDduHLRaLcLCwjB//nyn4ujvmEyRx2s2WpB7uRolN/ougRECOHChDMeLqtBktNi3B+m8EeDjDY1adSs+kwUD+HJvRNQHrOXVsOUXAwBkJlNEirMeOQXrkVMQJrN9myiphPXIKdi+vepY9suvYT1yCmg13SpbVtVW9tIVx7LHzrSVbW65VbbyhlMxNjU14aWXXsLJkydx6NAhyLKM+fPnw2azwWAwYMqUKSgrK8PevXtx5swZrFixArbvErd9+/Zh/vz5ePDBB/H111/j0KFDGD9+vFNx9Hcc5kcer7TWgJJaA2oMrYgN9oUsu77HqsVkQY2hFXEhbfMQZFlCsE4Dm03AbL31i1B8qB/iQ2/NVSira8Kp4hqkRQchLSrQ5XEREQGAqKwGzBbA3xdyYozS4RCRG8jOznb4+09/+hPCw8Nx4cIF/Otf/0J1dTVyc3MREhICAA6XiVi/fj0WLlyIdevW2bdlZGT0TeB9jMkUebzkyAA0tJiREhnQK4mUyWLFkW8rYWg1QwjYk6UpaVF3HWposthgttpQWd/cFl8/m+NFRJ5BXK0AAKiyhvJivUT9gOoHYwAAkurWSBUpPgqquEjgju8CqslZbXdu++xKgyKgiglvX3ZCRvuyUaFOxVhQUIA1a9bg+PHjqKmpsfc6lZSUIC8vD1lZWfZE6k55eXl49tlnnXpdd8NkijyeSpYxPim8157fSyUjKsAH5TaBUL9bi0p0Z85WYpgfVLLU1mPGRIqIeon1TD4Azpci6i9uT6Ls2zr5oaO3yt7N3LlzkZCQgD/+8Y+IiYmBzWbDyJEjYTKZ4OPj0+Vj77bfk/DnKRpwLFbnJmJ2RpIkpMeFYNqwGPhqvHr82PhQv17pMSMiAgDRYoQ177uV/DK4kh8R3d2NGzeQn5+PV155BTNmzMCwYcNQV1dn35+eno68vDzU1tZ2+Pj09HQcOnSor8JVFJMpGlCKqvT45Nw13DC03tPzNBnNuFheb184QpIkaL3a/xrUE0IIFFXpUVzTeE/PQ0R0O8uXpyHKqwFZgio9RelwiMgNBAcHIzQ0FO+++y4KCwtx+PBhvPTSS/b9P/7xjxEVFYVHHnkER48exeXLl7F79277Rc3Xrl2LHTt2YO3atbh48SLOnTuH1157Tanq9ComUzSg1DWbYLJYcaXa+YTFarPhi2+v40J5HS5VNLgstrL6ZuSV3EBeSS2ajOa7P4CI6C6EzWZf7UtOjofkq1M4IiJyB7IsY+fOnTh16hRGjhyJn/3sZ/jtb39r3+/t7Y39+/cjIiICDz74IEaNGoWNGzdC9d0ww6lTp+LDDz/E3r17kZmZienTp+PEiRNKVadXcc4UDSgjYoIQ4uuNxDB/+zYhRI+uSaWSZQyNDsS3lQ1ICPO7+wO6aVCQDtFBOoT7a6Hz5keTiO6dJMsQLUYAgGr0cIWjISJ3cv/99+PChQsO226/lEtCQgJ27drV6eMfffRRPProo70WX3/Bb2w0oPh4q5EUHmD/u7HVjKMF1zE4zA+pUYHdTqoSw/wRF+ILlQtXxZIkCROGRCh6sWEi8jzWvIsAAFUm50sREbkah/nRgHa1xoAmoxk1BqNDEnPnRXQbW83IvVztsHiFKxOpm26PwSYEGlpMXZQmIuqaMJlhO18EAFBlcCU/IiJXY88UDWhDowPh7+PlMKzOYrXh8MVyDAr2xdDott6qY4XX0dhqhlolIyvBues19ITRbMWxoiroW0yYMbznqwQSEQmTGebdBwCzBVJIIKT4KKVDIiLyOOyZogFNrZKREOqHcH+tfdu1uiY0tppxra4JsiRBliSMTghDqJ8Ww2KC+iQuL5UMIQQE2nrFiIh6StTpYbX3SqVxCDERUS9gzxTRHeJCfKH+bgjfzS8fYf5aTEmL6rMvI7IsYXxSOIQA/LTslSKinpOCAyBu1AMAZM6XIiLqFUymiO6gkmXEhvi2297Xv+reObTPZLHCW31v17IiogHESw3bxcsAAFXWMIWDISLyTBzmR+QGmoxm/PObMpwvq4PtjsUxiIg6IipqIKpqAbUKqhHJSodDROSRmEwRuYFrdc0wWayobmxVOpQBzWK14dvKBpy7VttuxUei/kTUN8LyeS4AQE5LhOSjUTgiIiLPxGF+RG4gLSoQvt5qBPt6Q/5uuGFPLzZM9664xoBz12oR4qtxeO8bWkzw13hBlnk8qH+wlVfD8sVpAIAqk0uiExH1FvZMEbmJ2BBfh3lUl6sbcaq4xuHaV9S7ksL9ERHgg8Hh/vZtNiFwJL8S/zhTAj2vC0b9hOTnA1tJBQAmU0TU9xITE/H73/9e6TD6BHumiNxQq9mKc9fqYLXZEOqnQWKY/90f1M+0mCwwGC0Oy9L3N5UNzbhW14wxCaGQJAmyLOG+VMdr9TQZLZAlwAbJYeXF8vpmWKw2RAfp4KXi71bUt6SIEAgmU0REvY7JFJEb0nqpMDE5Atdqm5AQ6qd0OD1W09iKz/MrofVSYXZ6rH3oYlGVHgE+Xgjz0yo+hLHVbMGxoirYbAIR/lrEd/I++2u9MDs9Ds1Gi70eAJBfUY/aJiNSIgORHhfSV2ETAQCs3xQCFiuk8GBIMeFKh0NE5LH4cymRm4oI8MHoxDB70mETAt9cq4PRbL3rY2ubjLh6w+AwLM0mBBqaTWgxWVy6uILVZkNZXROq9C32bcG+3vBWy/DVqO3xtpgsOFtaiyP5lahvVn64nNZLjeExwUiJDERMkK7LsrLk2CslhEBEgA/C/X367ELPRDeJViOseZcAtPVKKf3DBBHdIoSAaG5V5tbNtv3dd99FTEwMbDbHaQTz5s3DU089haKiIsybNw+RkZHw8/PDuHHjcPDgQaffk02bNmHUqFHw9fVFXFwcli9fDoPB4FDm6NGjmDp1KnQ6HYKDgzFz5kzU1dUBAGw2G37zm98gOTkZGo0G8fHxWL9+vdPx9BR7pog8xKWKeuRX1qO8vhn3j4iBLElobDHhyLfXIUvA7PQ4e9mC6w24VtuE9LgQBPh4A2gbOnjwQhkkScL80Qm3ylY2oFLfgsQwP8SFtPXOWG02lNxoglolITbY1/5l7eb8LZUs2bcVVTXi3LVahPlrERHg891+GTNHDnK4bpYkAYlh/jAYzQj2vbXymL7FBL8+WNyh2WTBudJajIwNts9NS4sKdOq5JEnCiEHB7bbnldyAv9YLSeH+/IJLvcZ68jwsh08A4BA/on6nxQjD2McUeWm/k38FdHcfWr9gwQK8+OKL+OyzzzBjxgwAQG1tLT799FN8/PHHMBgMePDBB7F+/XpoNBp88MEHmDt3LvLz8xEfH9/juGRZxv/8z/9g8ODBuHz5MpYvX44VK1bgrbfeAgDk5eVhxowZeOqpp7B582ao1Wp89tlnsFrbfoxduXIl/vjHP+KNN97A5MmTUVFRgUuXLvU4DmcxmSLyEDFBOlyrbcLQ6ED7cDOVLKPVbIEkSQ6r/wXpNDBZbPDxunUKsFpt0KhVkCTJ4Yt+fbMJVfoWRH6XCAGA0WLD6as1kKS2ZOqmb8rqUFSlx9DoIHsyERvsi6IqPUJ8NQ4x3HkBYq2XGlkJoQ6/nNlsAl8WXIcEYFJKpD3x6w15JTdQUd8Mi01gUkqky5+/prEVRVV6ABJC/TQI0nGpandUUFCAxYsXo6amBoGBgfjf//1fjBgxwqHM4cOH8Ytf/AIGgwGSJOGhhx7Cxo0bIcu9PxhEtBohLFbYikoBAKrMtF5/TSLyLMHBwZg9ezb+8pe/2JOpXbt2ISwsDNOmTYMsy8jIyLCX/+///m989NFH2Lt3L1544YUev95//ud/2u8nJibiV7/6FZYtW2ZPpn7zm99g7Nix9r8B2M+7jY2N2Lx5M958800sXrwYADBkyBBMnjy5x3E4i8kUkYcI0mkwY3gMVLd9YdN6qzB9WAw0dyQuaVGB7Xpd/H28MSczvt0wgKQIf4QHaB16iwAgOkgHIeCQeJktbT1Tty+4oNOoMWtUbLd7Ym4v12g0tw2JgARfza3T1fWGFsiyhGCdN9ROLO5QXNOIivpmpEYFItSv7Ve69NgQmK02jIxt36PkCqF+GmTGh6LFZGUi5caWLl2KJUuWICcnB7t27UJOTg5yc3MdygQHB2Pnzp1ISkpCa2sr7r//fnzwwQfIycnp9fgkrQby4EFAYxOgVkMePqTXX5OIesBH09ZDpNBrd9eiRYvw7LPP4q233oJGo8H27duxcOFCyLIMg8GAV199Ffv27UNFRQUsFgtaWlpQUlLiVFgHDx7Ehg0bcOnSJej1elgsFrS2tqK5uRk6nQ55eXlYsGBBh4+9ePEijEajPelTApMpIg+iuuOXb1mS2iVBd3Nn0hPqp7UnHDfpvNWYmNy+92bs4DBkJoTizrTJ2SFtgT7emDUqFo2tZoe6nSmtRWOrCZNSIhEV2DafydBqRnVjK4J03vY624RAfbMJ+haTw4qH1/UtKK9vRpBOY6+bn9YLU9KinYqzOyRJwpCIAIdtRrMVJ4trMDI2GIG92OtGrlFVVYWTJ09i//79AIDs7Gy88MILKCwsRHJysr1cVlaW/b5Wq0VmZiaKi4v7LE7b2W8BAPKIIZA0/L8i6k8kSerWUDulzZ07F0II7Nu3D+PGjcMXX3yBN954AwDw85//HAcOHMDrr7+O5ORk+Pj44Ic//CFMpp7Pdy4uLsacOXPw3HPPYf369QgJCcGXX36Jp59+GiaTCTqdDj4+Pp0+vqt9fYULUBCRy0iSBC+V7FRvUWdUsuzQkyOEgL9WDZ23GgHaW18UqxpbcPpqDc6X1dm3Gc1WfHaxHKeKb8BkubUwR0KoH0bGhiAmuOuFJXrb+bI6VDY04+SVGpcu+kG9o7S0FNHR0VCr236HlCQJ8fHxXf4aW1lZiV27dmHOnDmdljEajdDr9Q63e3Fr8QkO8SMi52i1Wjz66KPYvn07duzYgbS0NIwePRpA22IQOTk5mD9/PkaNGoWoqCinfzA6deoUbDYbfve73+H73/8+UlNTUV5e7lAmPT0dhw4d6vDxKSkp8PHx6XR/X2DPFBG5FUmSMKGDXjGtWoXIAB+E3XbdKh9vNUJ8NdB4qWC22uzztKICdXBybQmXGhoTBKPFirToIHvvXXl9MwquNyDijpUAr1Q3QkAgJsgXWq+2elisNtiEgJdK5oIW/ZBer8fcuXOxYsUKjB07ttNyGzZswLp161z2uta8fACAKoOLTxCR8xYtWoQ5c+bg/PnzePzxx+3bU1JSsGfPHsydOxeSJGH16tXtVv7rruTkZJjNZvzhD3/A3LlzcfToUbzzzjsOZVauXIlRo0Zh+fLlWLZsGby9vfHZZ59hwYIFCAsLw8svv4wVK1bA29sbkyZNQnV1Nc6fP4+nn376nurfXYr2TBUUFGDixIlITU3FuHHjcP78+Q7Lvf/++0hJScGQIUPw7LPPwmw2d2sfEQ0cMcG+mJwahaHRQQ7bpw2LwcTkSPsKff2JzluNCcmRCLltKGaT0YyaxlboWx2HS1wor8PXV2+g1Wyxbyurb8bf80rwZcF1h7JnSm7g9NUaGFpvnQ8bWkwovK5HeX2zQ9n8inqcLa1Fs/HW89Y3G3G2tBaXqx17SK7VNuFKdSOaTRYMRHFxcfb5AUBbL2lJSUmHq1c1NjZi1qxZmDdvHl566aUun3flypVoaGiw30pLS52OUTS1wPZtMQD2TBHRvZk+fTpCQkKQn5+Pn/zkJ/btmzZtQnBwMCZOnIi5c+di5syZ9l6rnsrIyMCmTZvw2muvYeTIkdi+fTs2bNjgUCY1NRX79+/HmTNnMH78eEyYMAH/93//Zx8lsHr1avzXf/0X1qxZg2HDhuGxxx5DVVWV8xXvIUkoOLZk+vTpeOKJJ+wTeV977bV2E3mvXLmCSZMm4fTp04iMjMS8efMwc+ZMPP/8813u6w69Xo/AwEA0NDQgICDg7g8gIuplhlYz6pqN0HqpEX5bL9vpqzUwmm3IjA+Bj3dbA1J4XY8zpTcQF+KH8Um3Lsz6j7wSGC1WzBgeYx8ieaW6Eaev1iA6SOcw3+3Tc9fQZDRj6tBo+/yxkhsG5F6pRri/D36QFmUve+B8GfQtJtyXGmVf5t5Z7nr+nTp1KnJycuzt1saNG3Hy5EmHMgaDATNnzsTMmTOxZs2aHr/Gvbw3luNn0fLkakhRYfA7/H6PX5uIXKe1tRVXrlzB4MGDodX2/3lSA0VXx8WZ869iPVM3J/Le7DbMzs5GaWkpCgsLHcrt2rULDz/8MKKioiBJEpYtW4YdO3bcdV9HXD0unYjI1fy0XogL8XNIpABgdEIYJiRH2BMpAEiODMD80YnISgh1KDt8UBCGxQRBd1tZX40ascG+7RYTSQzzQ2pUoH3oIAD4a72QGhWI2BDHOWUR/lpEB+mg8XJcHXIg2bJlC7Zs2YLU1FRs3LgRW7duBQA888wz2Lt3LwBg8+bNOHHiBPbs2YPMzExkZmb23QUkVSqoJmRAPTnr7mWJiOieKTZnqquJvLevilRSUoKEhFsXEE1MTLRP9u1qX0dcPS6diEhpsixBvmP9xKTw9r+mRQT4dNibdOewSAAI9tV0uApkRnxou20DTVpaGo4dO9Zu+3vvvWe/v2rVKqxataovw7JTjx0B9fu/VOS1iYjutH37dixdurTDfQkJCZ1O8XEnA2oBipUrVzqMXdfr9YiLi1MwIiIiIiIiz/Twww/je9/7Xof7vLz631xmZyiWTN0+kVetVnc6kTc+Ph5FRUX2v4uLi+1lutrXEY1GA42GF8skIiIiIupt/v7+8Pf3v3tBN6bYnKmIiAiMHj0a27ZtAwDs3r0bsbGxDkP8gLa5VHv37kVlZSWEEHjnnXewcOHCu+4jIiIiIlIaryPYv7j6eCi6NHp3JvImJSVh3bp1mDRpEpKTkxEeHm4fe9nVPiIiIiIipdwcxtbc3HyXktSXbh4PVw0zVHRpdKW569K8RETujuffzvG9IfIcFRUVqK+vR0REBHQ6HS+wriAhBJqbm1FVVYWgoCBER0e3K+PM+XdALUBBRERERNRXoqLartXXlxeRpa4FBQXZj4srMJkiIiIiIuoFkiQhOjoaERERMJvNSocz4Hl5eUGlcu21EplMERERERH1IpVK5fIv8dQ/KLoABRERERERkbtiMkVEREREROQEJlNEREREREROGNBzpm6uCq/X6xWOhIhoYLl53h3AV+foFNsmIiJlONM2DehkqrGxEQAQFxencCRERANTY2MjAgMDlQ6jX2HbRESkrJ60TQP6or02mw3l5eXw9/d36iJqer0ecXFxKC0t9bgLK7Ju7suT68e6uaeO6iaEQGNjI2JiYiDLHHF+O7ZNnWPd3JMn1w3w7PoNtLo50zYN6J4pWZYRGxt7z88TEBDgcf9gN7Fu7suT68e6uac768YeqY6xbbo71s09eXLdAM+u30CqW0/bJv4cSERERERE5AQmU0RERERERE5gMnUPNBoN1q5dC41Go3QoLse6uS9Prh/r5p48uW79kSe/36ybe/LkugGeXT/W7e4G9AIUREREREREzmLPFBERERERkROYTBERERERETmByRQREREREZETmEw5qaCgABMnTkRqairGjRuH8+fPKx2SSyUmJiItLQ2ZmZnIzMzEX//6V6VDctq///u/IzExEZIkIS8vz77dE45hZ3XzhOPX2tqKRx55BKmpqcjIyMC//du/obCwEABQVVWFWbNmISUlBSNHjsSRI0cUjrZnuqrb1KlTMXjwYPuxe+ONNxSOtuceeOABpKenIzMzE/fddx++/vprAJ7xmevvPPk99oTz2k2e3C4Bnts2eXK7BLBtcvpzJ8gp06ZNE1u3bhVCCPHhhx+KsWPHKhuQiyUkJIivv/5a6TBc4vPPPxelpaXt6uQJx7CzunnC8WtpaRH79u0TNptNCCHEH/7wBzFlyhQhhBBPPvmkWLt2rRBCiBMnTohBgwYJk8mkUKQ911XdpkyZIj766CPlgnOBuro6+/09e/aI9PR0IYRnfOb6O09+jz3hvHaTJ7dLQnhu2+TJ7ZIQbJuEcO5zx2TKCdevXxf+/v7CbDYLIYSw2WwiMjJSFBQUKByZ67j7Ca8jt9fJ046hpzVYHcnNzRUJCQlCCCF8fX1FRUWFfd+4cePEgQMHFIrs3t1eN09osG63detWkZGR4XGfuf7I099jTzyveXK7JITnt02e3C4JwbapuzjMzwmlpaWIjo6GWq0GAEiShPj4eJSUlCgcmWs98cQTGDVqFJ5++mlUV1crHY5LDYRj6GnHb/PmzZg3bx5u3LgBs9mMqKgo+77ExES3PnY363bTL37xC4waNQqPPfYYLl++rGBkznviiScQFxeH1atX489//vOA+MwpbSC8x552XrvdQDh+gGcdQ09ulwC2Td3FZIo6dOTIEZw9exanT59GWFgYFi9erHRI1AOedvx+/etfo7CwEBs2bFA6FJe7s25//vOfcenSJZw9exb33Xcf5syZo3CEzvnggw9QWlqKX/3qV3j55ZeVDoc8gKed1wYiTzqGntwuAWybeqS3us88mSd2xXelvLxc+Pn5KR3GPfPk4RRdDZ1w9+P329/+VowZM8ZhrLNOp/OI4RQd1e1OGo1G1NTU9F1QvUCr1YrKykqP+sz1R552XuuKu5/XbvLkdkkIz22bPLldEoJtE4f59YGIiAiMHj0a27ZtAwDs3r0bsbGxSE5OVjgy12hqakJ9fb397x07diArK0u5gHqBJx9DTzp+mzZtwo4dO3DgwAEEBQXZty9YsADvvPMOACA3NxdlZWWYMmWKQlE6p6O6WSwWXL9+3V5m9+7diIyMRGhoqEJR9lx9fT3Ky8vtf//tb39DaGioR3/m+gtPfo896bzWGU8+foDnHENPbpcAtk3OfO4kIYRwecQDQH5+PnJycnDjxg0EBARg69atGDVqlNJhucTly5eRnZ0Nq9UKIQSSkpKwefNmJCYmKh2aU5YuXYp9+/ahsrISoaGh8Pf3R2FhoUccw47qtn//fo84fteuXUNcXBySkpLg7+8PANBoNDh+/DiuX7+On/70p7hy5Qq8vb3x5ptvYtq0aQpH3H2d1e3w4cOYMmUKjEYjZFlGWFgYNm3ahIyMDIUj7r6rV69iwYIFaGlpgSzLCA8Px+uvv47MzEyP+Mz1d576HrNdci+e2jZ5crsEsG1y9nPHZIqIiIiIiMgJHOZHRERERETkBCZTRERERERETmAyRURERERE5AQmU0RERERERE5gMkVEREREROQEJlNEREREREROYDJFRERERETkBCZTRAOAJEn429/+pnQYREREANgukedgMkXUy3JyciBJUrvbrFmzlA6NiIgGILZLRK6jVjoAooFg1qxZ2Lp1q8M2jUajUDRERDTQsV0icg32TBH1AY1Gg6ioKIdbcHAwgLahDm+//TZmz54NHx8fJCUlYdeuXQ6PP3fuHKZPnw4fHx+EhoZiyZIlMBgMDmX+9Kc/YcSIEdBoNIiOjsYLL7zgsL+mpgbz58+HTqdDSkoK9u7da99XV1eHRYsWITw8HD4+PkhJSWnXyBIRkedgu0TkGkymiPqB1atXIzs7G2fOnMGiRYuwcOFCXLx4EQDQ1NSEmTNnIjg4GLm5ufjwww9x8OBBh0bp7bffxvPPP48lS5bg3Llz2Lt3L5KTkx1eY926dfjRj36Es2fP4sEHH8SiRYtQW1trf/0LFy7gk08+wcWLF/H2228jLCys794AIiLqV9guEXWTIKJetXjxYqFSqYSvr6/Dbf369UIIIQCIZcuWOTzme9/7nnjuueeEEEK8++67Ijg4WBgMBvv+ffv2CVmWRWVlpRBCiJiYGLFq1apOYwAgXnnlFfvfBoNBABCffPKJEEKIuXPniieffNI1FSYion6N7RKR63DOFFEfmDZtGt5++22HbSEhIfb7EyZMcNg3YcIE5OXlAQAuXryIjIwM+Pr62vdPmjQJNpsN+fn5kCQJ5eXlmDFjRpcxpKen2+/7+voiICAAVVVVAIDnnnsO2dnZOH36NB544AE88sgjmDhxolN1JSKi/o/tEpFrMJki6gO+vr7thje4io+PT7fKeXl5OfwtSRJsNhsAYPbs2bh69So+/vhjHDhwADNmzMDzzz+P119/3eXxEhGR8tguEbkG50wR9QNfffVVu7+HDRsGABg2bBjOnDmDpqYm+/6jR49ClmWkpaXB398fiYmJOHTo0D3FEB4ejsWLF2Pbtm34/e9/j3ffffeeno+IiNwX2yWi7mHPFFEfMBqNqKysdNimVqvtk2k//PBDjB07FpMnT8b27dtx4sQJvP/++wCARYsWYe3atVi8eDFeffVVVFdX48UXX8RPf/pTREZGAgBeffVVLFu2DBEREZg9ezYaGxtx9OhRvPjii92Kb82aNRgzZgxGjBgBo9GIf/zjH/ZGk4iIPA/bJSLXYDJF1Ac+/fRTREdHO2xLS0vDpUuXALStaLRz504sX74c0dHR2LFjB4YPHw4A0Ol0+Oc//4n/+I//wLhx46DT6ZCdnY1NmzbZn2vx4sVobW3FG2+8gZ///OcICwvDD3/4w27H5+3tjZUrV6K4uBg+Pj647777sHPnThfUnIiI+iO2S0SuIQkhhNJBEA1kkiTho48+wiOPPKJ0KERERGyXiHqAc6aIiIiIiIicwGSKiIiIiIjICRzmR0RERERE5AT2TBERERERETmByRQREREREZETmEwRERERERE5gckUERERERGRE5hMEREREREROYHJFBERERERkROYTBERERERETmByRQREREREZET/h9bOr9Xp7HRPQAAAABJRU5ErkJggg=="/>
