---
layout: page
title: "torchtext를 활용한 텍스트 데이터 전처리 방법"
description: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
headline: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-18
---

`torchtext`는 pytorch 모델에 주입하기 위한 텍스트 데이터셋을 구성하기 편하게 만들어 주는 데이터 로더(Data Loader) 입니다. `torchtext` 를 활용하여 CSV, TSV, JSON 등의 정형 데이터셋을 쉽게 로드하도록 도와주는 `TabularDataset` 클래스의 활용 방법과 제공해주는 토크나이저(tokenizer) 워드 벡터(Word Vector) 를 적용하는 방법에 대하여 알아보겠습니다.

튜토리얼의 끝 부분에는 Pandas 의 DataFrame을 Data Loader 로 쉽게 변환하는 방법도 알아보도록 하겠습니다.

예제 코드는 아래에서 확인할 수 있습니다.

**예제 코드**
[Google Colab 예제 코드](https://colab.research.google.com/github/teddylee777/machine-learning/blob/master/02-PyTorch/16-torchtext-tutorial.ipynb)


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


# torchtext 튜토리얼


## 샘플 데이터셋 다운로드



```python
import urllib

url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
urllib.request.urlretrieve(url, 'bbc-text.csv')
```

<pre>
('bbc-text.csv', <http.client.HTTPMessage at 0x7fef4303e940>)
</pre>
Pandas로 데이터 로드 및 출력



```python
import pandas as pd

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


## 토크나이저 생성



```python
from torchtext.data.utils import get_tokenizer
```

tokenizer의 타입으로는 `basic_english`, `spacy`, `moses`, `toktok`, `revtok`, `subword` 이 있습니다.



다만, 이 중 몇개의 타입은 추가 패키지가 설치되어야 정상 동작합니다.



```python
tokenizer = get_tokenizer('basic_english', language='en')
tokenizer("I'd like to learn torchtext")
```

<pre>
['i', "'", 'd', 'like', 'to', 'learn', 'torchtext']
</pre>
토큰 타입을 지정하면 그에 맞는 tokenizer를 반환하는 함수를 생성한 뒤 원하는 타입을 지정하여 tokenizer를 생성할 수 있습니다.



```python
def generate_tokenizer(tokenizer_type, language='en'):
    return get_tokenizer(tokenizer_type, language=language)
```

`basic_english`를 적용한 경우



```python
tokenizer = generate_tokenizer('basic_english')
tokenizer("I'd like to learn torchtext")
```

<pre>
['i', "'", 'd', 'like', 'to', 'learn', 'torchtext']
</pre>
`toktok`을 적용한 경우



```python
tokenizer = generate_tokenizer('toktok')
tokenizer("I'd like to learn torchtext")
```

<pre>
['I', "'", 'd', 'like', 'to', 'learn', 'torchtext']
</pre>

```python
from nltk.tokenize import word_tokenize

word_tokenize("I'd like to learn torchtext")
```

<pre>
['I', "'d", 'like', 'to', 'learn', 'torchtext']
</pre>
## 필드(Field) 정의



```python
from torchtext.legacy import data
```

`torchtext.legacy.data.Field` 

- `Field` 클래스는 `Tensor`로 변환하기 위한 지침과 함께 데이터 유형을 정의합니다. 

- `Field` 객체는 `vocab` 개체를 보유합니다.

- `Field` 객체는 토큰화 방법, 생성할 Tensor 종류와 같이 데이터 유형을 수치화하는 역할을 수행합니다.



```python
TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자 화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)
```

`fields` 변수에 dictionary를 생성합니다.

- `key`: 읽어 들여올 파일의 열 이름을 지정합니다.

- `value`: (`문자열`, `data.Field`) 형식으로 지정합니다. 여기서 지정한 문자열이 나중에 생성된 data의 변수 이름으로 생성됩니다.



(참고) fields에 `[('text', TEXT), ('label', LABEL)]` 와 같이 생성하는 경우도 있습니다. 컬러명 변경이 필요하지 않은 경우는 `List(tuple(컬럼명, 변수))`로 생성할 수 있습니다.



```python
fields = {
    'text': ('text', TEXT), 
    'category': ('label', LABEL)
}
```

## 데이터셋 로드 및 분할


`TabularDataset` 클래스는 정형 데이터파일로부터 직접 데이터를 읽을 때 유용합니다.



지원하는 파일 형식은 `CSV`, `JSON`, `TSV` 을 지원합니다.



```python
import random
from torchtext.legacy.data import TabularDataset

SEED = 123

dataset = TabularDataset(path='bbc-text.csv',  # 파일의 경로
                         format='CSV',         # 형식 지정
                         fields=fields,        # 이전에 생성한 field 지정
                         skip_header=True    # 첫 번째 행은 컬러명이므로 skip
                        )        
```

이전에 생성한 `dataset` 변수로 train / test 데이터셋을 분할 합니다.



```python
train_data, test_data = dataset.split(split_ratio=0.8,               # 분할 비율
                                      stratified=True,               # stratify 여부
                                      strata_field='label',          # stratify 대상 컬럼명
                                      random_state=random.seed(SEED) # 시드
                                     )
```


```python
# 생성된 train / test 데이터셋의 크기를 출력 합니다.
len(train_data), len(test_data)
```

<pre>
(1781, 444)
</pre>
## 단어 사전 생성



```python
TEXT.build_vocab(train_data, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_data)
```


```python
NUM_VOCABS = len(TEXT.vocab.stoi)
NUM_VOCABS
```

<pre>
1002
</pre>

```python
TEXT.vocab.freqs.most_common(10)
```

<pre>
[('the', 41674),
 ('to', 19644),
 ('of', 15674),
 ('and', 14621),
 ('a', 14327),
 ('in', 13995),
 ('s', 7126),
 ('for', 7054),
 ('is', 6535),
 ('that', 6329)]
</pre>
`TEXT.vocab.stoi`는 문자열을 index로, `TEXT.vocab.itos`는 index를 문자열로 변환합니다.



```python
TEXT.vocab.stoi
```

```
defaultdict({'<unk>': 0,
             '<pad>': 1,
             'the': 2,
             'to': 3,
             'of': 4,
             'and': 5,
             ...
             'dems': 995,
             'laws': 996,
             'rival': 997,
             'story': 998,
             'watch': 999,
             ...})
```

```python
# string to index
print(TEXT.vocab.stoi['this'])
print(TEXT.vocab.stoi['pretty'])
print(TEXT.vocab.stoi['original'])

print('==='*10)

# index to string
print(TEXT.vocab.itos[14])
print(TEXT.vocab.itos[194])
print(TEXT.vocab.itos[237])
```

```
37
0
849
==============================
was
end
record
```

## 버킷 이터레이터 생성



- `BucketIterator` 의 주된 역할은 데이터셋에 대한 배치 구성입니다.



```python
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),     # dataset
    sort=False,
    repeat=False,
    batch_size=BATCH_SIZE,       # 배치사이즈
    device=device)               # device 지정
```

1개의 배치를 추출합니다.



```python
# 1개의 batch 추출
sample_data = next(iter(train_iterator))
```

`text` 의 shape 를 확인합니다.



```python
# batch_size, sequence_length
sample_data.text.shape
```

<pre>
torch.Size([32, 120])
</pre>

```python
len(sample_data.text)
```

<pre>
32
</pre>

```python
sample_data.label.size(0)
```

<pre>
32
</pre>
`label` 의 shape 를 확인합니다.



```python
# batch_size
sample_data.label.shape
```

<pre>
torch.Size([32])
</pre>

```python
# label을 출력합니다.
sample_data.label
```

<pre>
tensor([5, 1, 2, 4, 1, 4, 5, 2, 2, 4, 1, 2, 5, 3, 1, 3, 4, 4, 1, 4, 3, 3, 2, 1,
        3, 5, 2, 4, 1, 5, 3, 5], device='cuda:1')
</pre>
아래에서 확인할 수 있듯이 `<unk>` 토큰 때문에 카테고리의 개수가 5개임에도 불구하고 index는 0번부터 5번까지 맵핑되어 있습니다.



```python
LABEL.vocab.stoi
```

```
defaultdict({'<unk>': 0,
             'sport': 1,
             'business': 2,
             'politics': 3,
             'tech': 4,
             'entertainment': 5})
```
따라서, 0번을 무시해주기 위해서는 배치 학습시 다음과 같이 처리해 줄 수 있습니다.



1을 subtract 해줌으로써 0~4번 index로 조정해 주는 것입니다.



```python
sample_data.label.sub_(1)
```

<pre>
tensor([4, 0, 1, 3, 0, 3, 4, 1, 1, 3, 0, 1, 4, 2, 0, 2, 3, 3, 0, 3, 2, 2, 1, 0,
        2, 4, 1, 3, 0, 4, 2, 4], device='cuda:1')
</pre>
## 데이터프레임(DataFrame) 커스텀 데이터셋 클래스



`torchtext.legacy.data.Dataset`을 확장하여 DataFrame을 바로 `BucketIterator`로 변환할 수 있습니다.



```python
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123

# 데이터프레임을 로드 합니다.
df = pd.read_csv('bbc-text.csv')

# 컬럼명은 text / label 로 변경합니다
df = df.rename(columns={'category': 'label'})
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
      <th>label</th>
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



```python
# train / validation 을 분할 합니다.
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
```


```python
# train DataFrame
train_df.head()
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
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1983</th>
      <td>sport</td>
      <td>officials respond in court row australian tenn...</td>
    </tr>
    <tr>
      <th>878</th>
      <td>tech</td>
      <td>slow start to speedy net services faster broad...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>politics</td>
      <td>amnesty chief laments war failure the lack of ...</td>
    </tr>
    <tr>
      <th>1808</th>
      <td>sport</td>
      <td>dal maso in to replace bergamasco david dal ma...</td>
    </tr>
    <tr>
      <th>1742</th>
      <td>tech</td>
      <td>technology gets the creative bug the hi-tech a...</td>
    </tr>
  </tbody>
</table>
</div>



```python
# validation DataFrame
val_df.head()
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
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>717</th>
      <td>politics</td>
      <td>child access laws shake-up parents who refuse ...</td>
    </tr>
    <tr>
      <th>798</th>
      <td>entertainment</td>
      <td>fry set for role in hitchhiker s actor stephen...</td>
    </tr>
    <tr>
      <th>1330</th>
      <td>business</td>
      <td>palestinian economy in decline despite a short...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>business</td>
      <td>japanese banking battle at an end japan s sumi...</td>
    </tr>
    <tr>
      <th>1391</th>
      <td>business</td>
      <td>manufacturing recovery  slowing  uk manufactur...</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 필요한 모듈 import
import torch
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer

# device 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
```

<pre>
cuda:1
</pre>
`torchtext.legacy.data.Dataset`을 상속하여 데이터프레임을 로드할 수 있습니다.



```python
class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            # text, label 컬럼명은 필요시 변경하여 사용합니다
            label = row['label'] if not is_test else None
            text = row['text'] 
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, False, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
```


```python
# 토크나이저 정의 (다른 토크나이저로 대체 가능)
tokenizer = get_tokenizer('basic_english')
```

앞선 내용과 마찬가지로 `Field`를 구성합니다.



```python
TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)

# fiels 변수에 List(tuple(컬럼명, 변수)) 형식으로 구성 후 대입
fields = [('text', TEXT), ('label', LABEL)]
```


```python
# DataFrame의 Splits로 데이터셋 분할
train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=val_df)
```


```python
# 단어 사전 생성
TEXT.build_vocab(train_ds, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_ds)
```


```python
# 단어 사전 개수 출력
NUM_VOCABS = len(TEXT.vocab)
NUM_VOCABS
# 개수 1000 + <unk> + <pad> : 총 1002개
```

<pre>
1002
</pre>
`BucketIterator`를 생성합니다.



```python
BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_ds, val_ds), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)
```


```python
# 1개 배치 추출
sample_data = next(iter(train_iterator))
```


```python
# text shape 출력 (batch_size, sequence_length)
sample_data.text.shape
```

<pre>
torch.Size([32, 120])
</pre>

```python
# label 출력 (batch)
sample_data.label
```

<pre>
tensor([1, 2, 4, 4, 3, 4, 5, 4, 5, 1, 2, 1, 2, 2, 5, 5, 2, 5, 5, 2, 5, 1, 1, 2,
        5, 5, 1, 3, 2, 3, 3, 5], device='cuda:1')
</pre>