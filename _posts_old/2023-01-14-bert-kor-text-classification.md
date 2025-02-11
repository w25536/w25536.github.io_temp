---
layout: page
title: "[huggingface] 한글 pre-trained 사전학습 BERT 모델로 텍스트 분류하기"
description: "[huggingface] 한글 pre-trained 사전학습 BERT 모델로 텍스트 분류하기에 대해 알아보겠습니다."
headline: "[huggingface] 한글 pre-trained 사전학습 BERT 모델로 텍스트 분류하기에 대해 알아보겠습니다."
categories: huggingface
tags: [python, 파이썬, huggingface, nsmc, 네이버 댓글, 한글 자연어처리, 자연어처리, 텍스트 분류, BERT, 허깅페이스, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-13
---

이번 포스팅에서는 Huggingface의 한글 데이터셋으로 사전 학습된  `kykim/bert-kor-base` 모델을 가져와서 한글 자연어 데이터에 대한 텍스트 분류기를 만들어 보겠습니다. 이미지넷에서는 전이학습을 통해 손쉽게 Transfer Learning을 수행할 수 있습니다. 하지만, Huggingface에 이제 막 입문 하신 분들은 구조에 대한 이해가 선행되어야 하기 때문에 이미지넷 모델 보다는 사전 학습 모델을 가져다 쓰기가 좀 더 어렵게 느낄 수 있습니다.

따라서, 이번 예제에서는 한글 데이터셋을 로드해서 사전 학습된 토크나이저(Tokenizer)로 문장에 대한 전처리, 데이터셋 구성, 배치 구성, 모델의 출력층 추가, Fine-tuning 까지 진행해 보도록 하겠습니다. 먼저, 사전 학습 모델을 공유해 주신 **kiyoung kim**님께 감사드립니다.

사전 학습 모델은 https://huggingface.co/kykim/bert-kor-base 에서 확인할 수 있습니다.

또한, 김기영님께서 YouTube에 [자연어처리 기초 강의](https://www.youtube.com/watch?v=Z201jwWo-xs&list=PLrLEKGJAgXxL-R9IqDH7HANWXRsS900tF)도 업로드 해주셨습니다. 한 번 들어보시면 유익하실 만한 내용이 많습니다!



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

## 데이터셋 로드 (Korpora)

`Korpora`는 한글 자연처리를 위한 데이터셋을 손쉽게 로드 해주도록 도와주는 오픈소스 파이썬 라이브러리 입니다.

- 링크: https://github.com/ko-nlp/Korpora



```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 한글 자연어 처리 데이터셋
from Korpora import Korpora

# 토크나이저 관련 경고 무시하기 위하여 설정
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# device 지정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

corpus = Korpora.load("nsmc")
```

<pre>
사용 디바이스: cuda:1
</pre>

<pre>

    Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
    손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.
    
    말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
    해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
    해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.
    
    # Description
    Author : e9t@github
    Repository : https://github.com/e9t/nsmc
    References : www.lucypark.kr/docs/2015-pyconkr/#39
    
    Naver sentiment movie corpus v1.0
    This is a movie review dataset in the Korean language.
    Reviews were scraped from Naver Movies.
    
    The dataset construction is based on the method noted in
    [Large movie review dataset][^1] from Maas et al., 2011.
    
    [^1]: http://ai.stanford.edu/~amaas/data/sentiment/
    
    # License
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
    Details in https://creativecommons.org/publicdomain/zero/1.0/

</pre>

본 튜토리얼은 `Korpora`의 **네이버 영화 댓글 데이터셋**으로 진행합니다.


- github 주소: https://github.com/e9t/nsmc



```python
import pandas as pd

train = pd.read_csv('~/Korpora/nsmc/ratings_train.txt', sep='\t')
test = pd.read_csv('~/Korpora/nsmc/ratings_test.txt', sep='\t')
```

문장의 길이를 계산하여 `length` 컬럼에 담습니다.



```python
train['length'] = train['document'].apply(lambda x: len(str(x)))
test['length'] = test['document'].apply(lambda x: len(str(x)))
```

문장의 길이가 단어 기준으로 **5 이상**인 문장만 가져옵니다. 너무 짧은 문장은 식별하기 어려워 제외토록 합니다.



```python
train = train.loc[train['length'] > 5]
# 전체 데이터셋 크기가 커서 1000개의 문장을 샘플링 합니다.
train = train.sample(1000)
train
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
      <th>id</th>
      <th>document</th>
      <th>label</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73093</th>
      <td>9626744</td>
      <td>킬링타임용도 아까움. .. 외계인 불쌍함. .. 연기도 전부 개판. . 내용개판. ..</td>
      <td>0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>39458</th>
      <td>8632842</td>
      <td>관람하면서 몸을 움직이고 싶어지는 영화</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>47733</th>
      <td>8030303</td>
      <td>절대 10점을 쥬지않는다</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>84687</th>
      <td>10075627</td>
      <td>이 영화는 대체 몇 번 째인지...봐도봐도 또 보고싶은 영화!</td>
      <td>1</td>
      <td>34</td>
    </tr>
    <tr>
      <th>75734</th>
      <td>7916982</td>
      <td>짝 퉁 쓰 레 기</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114578</th>
      <td>8618070</td>
      <td>기대는 안했는데 가족이란? 생각나게 하는 매력적인 영화입니다.</td>
      <td>1</td>
      <td>34</td>
    </tr>
    <tr>
      <th>34309</th>
      <td>8832611</td>
      <td>그나이에 전혀 현실감각 없는 이야기 소설책 부터 허접하더니 영화는 쩝...</td>
      <td>0</td>
      <td>41</td>
    </tr>
    <tr>
      <th>41251</th>
      <td>4377580</td>
      <td>이연걸의 액션이 멋진 영화</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>24006</th>
      <td>3895859</td>
      <td>정치에 있어서 소신의 중요함을 다시 보게 되었다</td>
      <td>1</td>
      <td>26</td>
    </tr>
    <tr>
      <th>103967</th>
      <td>4966212</td>
      <td>미드는 시즌이 지나면 개판된다는 진리를 보여줌. 시즌1최고, 2아침드라마급, 3쓰레기</td>
      <td>0</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>



```python
# test 데이터셋에도 동일하게 적용합니다.
test = test.loc[test['length'] > 5]
test = test.sample(500)
test
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
      <th>id</th>
      <th>document</th>
      <th>label</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14218</th>
      <td>4437078</td>
      <td>연기가 자연스럽지못하고, 지루했다</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>36410</th>
      <td>9990755</td>
      <td>굿굿재밌게봤어요~^^</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>17343</th>
      <td>9964083</td>
      <td>몰입도 안되고 이도 저도 아닌 유치하기 짝이없네</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>40714</th>
      <td>1599945</td>
      <td>내 인생 최고의 영화... 너무도 잔잔하고 예쁜 영화...</td>
      <td>1</td>
      <td>32</td>
    </tr>
    <tr>
      <th>24556</th>
      <td>10516</td>
      <td>정말 닮고 싶은 캐릭터네요.. 잘된 애니 같아여..^^</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38679</th>
      <td>8680926</td>
      <td>야근 시트콤임 지루함 밥먹는거 밖에 안나옴</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23279</th>
      <td>8154982</td>
      <td>두배우의 명성에 먹칠을하는 영화임</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>30633</th>
      <td>5780174</td>
      <td>스토리가 만화같다 연기또한.,.,</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>42485</th>
      <td>8763674</td>
      <td>진심 1시간 개똥만 보다가 날렸네</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>21779</th>
      <td>10164582</td>
      <td>후속작은 만들지 말아야 했다. 충분히 전편에서 주제를 심도깊게 던졌다. 그래서 영화...</td>
      <td>0</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>
</div>


## 토큰화가 적용된 데이터셋



먼저 `huggingface` 에서 한글 데이터셋으로 사전학습된 모델을 지정합니다.



이번에는 `'kykim/bert-kor-base'`을 사용할 예정이며, 관련 내용은 아래의 링크에서 확인할 수 있습니다.



- 링크: https://huggingface.co/kykim/bert-kor-base



```python
CHECKPOINT_NAME = 'kykim/bert-kor-base'
```


```python
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
  
    def __init__(self, dataframe, tokenizer_pretrained):
        # sentence, label 컬럼으로 구성된 데이터프레임 전달
        self.data = dataframe        
        # Huggingface 토크나이저 생성
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['document']
        label = self.data.iloc[idx]['label']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)           # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        # input_ids, attention_mask, token_type_ids 이렇게 3가지 요소를 반환하도록 합니다.
        # input_ids: 토큰
        # attention_mask: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)
        # token_type_ids: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)
    
```

`nn.Dataset`을 확장한 클래스인 `TokenDataset` 클래스 인스턴스를 생성합니다.



```python
# 토크나이저 지정
tokenizer_pretrained = CHECKPOINT_NAME

# train, test 데이터셋 생성
train_data = TokenDataset(train, tokenizer_pretrained)
test_data = TokenDataset(test, tokenizer_pretrained)

# DataLoader로 이전에 생성한 Dataset를 지정하여, batch 구성, shuffle, num_workers 등을 설정합니다.
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)
```

생성된 `train_loader`로부터 1개의 batch 만 꺼내서 값을 출력해 보겠습니다.



```python
# 1개의 batch 꺼내기
inputs, labels = next(iter(train_loader))

# 데이터셋을 device 설정
inputs = {k: v.to(device) for k, v in inputs.items()}
labels.to(device)
```

<pre>
tensor([1, 1, 0, 1, 0, 0, 1, 1], device='cuda:1')
</pre>
inputs 는 입력으로 들어가는 `x` 데이터 입니다.



각각의 key 에 대한 설명은 다음과 같습니다.



- `input_ids`: 토큰

- `attention_mask`: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)

- `token_type_ids`: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0



```python
# 생성된 inputs의 key 값 출력
inputs.keys()
```

<pre>
dict_keys(['input_ids', 'attention_mask', 'token_type_ids'])
</pre>
첫 번째 **8**은 `batch_size` 입니다.



그렇다면 두 번째 `512`는 의미하는 바가 뭘까요?



```python
# key 별 shape 확인
inputs['input_ids'].shape, inputs['attention_mask'].shape, inputs['token_type_ids'].shape
```

<pre>
(torch.Size([8, 512]), torch.Size([8, 512]), torch.Size([8, 512]))
</pre>
**512**에 대한 답변은 아래 `BertConfig`를 통해 확인할 수 있습니다.



아래 `"max_position_embeddings"` 값에 **512**가 할당되어 있는 것을 볼 수 있는데, 1개 문장의 길이가 최대 512개의 단어로 이루어지도록 embedding 되었다는 의미입니다. 즉, 1개 문장의 길이가 **512** 단어로 토크나이저된 결과이며, `batch_size`가 **8** 이기 때문에 결과 shape가 `[8, 512]`로 나오게 되었습니다.



```python
from transformers import BertConfig

config = BertConfig.from_pretrained(CHECKPOINT_NAME)
config
```

<pre>
BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "embedding_size": 768,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.22.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 42000
}
</pre>
`labels`는 one-hot encoding이 되어 있지 않은 상태로 출력됩니다.



```python
# labels 출력
labels
```

<pre>
tensor([1, 1, 0, 1, 0, 0, 1, 1])
</pre>
## 사전학습(pre-trained) Model 생성



사전학습 BERT 모델은 `BertModel`의 `from_pretrained` 함수로 간단히 가져올 수 있습니다.



GPU 학습을 위하여 `.to(device)` 도 설정하였습니다.



```python
from transformers import BertModel

# 모델 생성
model_bert = BertModel.from_pretrained(CHECKPOINT_NAME).to(device)
model_bert
```

<pre>
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(42000, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
</pre>

이전에 생성한 `inputs`를 한 번 대입해본 후 결과를 확인해 보도록 하겠습니다.



```python
output = model_bert(**inputs)
output.keys()
```

<pre>
odict_keys(['last_hidden_state', 'pooler_output'])
</pre>
결과를 `output` 에 담은 후 key를 출력해 보면 `'last_hidden_state'` 와 `'pooler_output'`이렇게 2개가 출력됩니다.



- `last_hidden_state`는 배치의 각 시퀀스에서 각 토큰에 대한 숨겨진 표현을 포함합니다. 따라서 크기는 `(batch_size, seq_len, hidden_size)`입니다.

- `pooler_output`은 배치의 각 시퀀스의 "표현"을 포함하며 크기`(batch_size, hidden_size)`입니다. 기본적으로 하는 일은 배치(hidden_size 크기의 벡터)에서 각 시퀀스의 `[CLS]` 토큰의 숨겨진 표현을 가져온 다음 BertPooler를 통해 실행하는 것입니다. 이것은 선형 레이어와 Tanh 활성화 함수로 구성됩니다.



```python
output['last_hidden_state'].shape, output['pooler_output'].shape
```

<pre>
(torch.Size([8, 512, 768]), torch.Size([8, 768]))
</pre>



```python
# last_hidden_state 출력
last_hidden_state = output['last_hidden_state']
print(last_hidden_state.shape)
print(last_hidden_state[:, 0, :])
```


<pre>
torch.Size([8, 512, 768])
tensor([[-0.2648, -0.7522, -0.0201,  ...,  0.1347, -1.0979,  0.5044],
[-0.8496,  0.3822, -0.1804,  ...,  0.0605, -1.2255,  0.1880],
[ 0.1074,  0.3704, -0.4008,  ..., -0.0877, -0.9114,  0.2701],
...,
[ 0.5518,  0.0208,  0.2635,  ...,  0.5427, -2.0081,  0.9168],
[ 0.8714, -0.0267,  0.2738,  ...,  0.1526, -1.0668,  0.5521],
[ 0.1562,  0.0058, -0.0917,  ..., -1.7647,  0.1226, -0.9639]],
device='cuda:1', grad_fn=SliceBackward0)
</pre>



```python
# pooler_output 출력
pooler_output = output['pooler_output']
print(pooler_output.shape)
print(pooler_output)
```

<pre>
torch.Size([8, 768])
tensor([[-0.1134,  0.3340, -0.9995,  ...,  0.1017,  0.2376,  0.7940],
        [ 0.5536,  0.3353, -0.9998,  ...,  0.7139,  0.7452, -0.9895],
        [-0.9024, -0.0207, -0.7576,  ..., -0.9268,  0.2992,  0.9743],
        ...,
        [-0.5301,  0.6815, -0.8834,  ...,  0.1480,  0.3735,  0.5666],
        [-0.9325, -0.0130, -0.7675,  ..., -0.6921,  0.0119,  0.8636],
        [ 0.6517, -0.2224, -0.9989,  ...,  0.9627,  0.3042, -0.4665]],
       device='cuda:1', grad_fn=TanhBackward0)
</pre>

output으로 나온 `last_hidden_state[:, 0, :]`의 `[CLS]` 토큰을 가져온 뒤 FC에 입력으로 대입합니다.



Binary-Classificition을 위하여 최종 출력 값은 **2** 로 설정합니다.



결과 shape은 `[8, 2]` 입니다. **8개의 batch** 에 대해 2개의 출력이 나왔습니다.



```python
fc = nn.Linear(768, 2)
fc.to(device)
fc_output = fc(last_hidden_state[:, 0, :])
print(fc_output.shape)
print(fc_output.argmax(dim=1))
```

<pre>
torch.Size([8, 2])
tensor([1, 0, 1, 1, 1, 1, 1, 1], device='cuda:1')
</pre>

## 사전학습(pre-trained) BERT 모델을 활용한 텍스트 분류 모델 생성



```python
class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        # 부모클래스 초기화
        super(CustomBertModel, self).__init__()
        # 사전학습 모델 지정
        self.bert = BertModel.from_pretrained(bert_pretrained)
        # dropout 설정
        self.dr = nn.Dropout(p=dropout_rate)
        # 최종 출력층 정의
        self.fc = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 입력을 pre-trained bert model 로 대입
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 결과의 last_hidden_state 가져옴
        last_hidden_state = output['last_hidden_state']
        # last_hidden_state[:, 0, :]는 [CLS] 토큰을 가져옴
        x = self.dr(last_hidden_state[:, 0, :])
        # FC 을 거쳐 최종 출력
        x = self.fc(x)
        return x
```

위의 정의한 `CustomBertModel` 클래스 인스턴스를 생성합니다.



```python
# CustomBertModel 생성
bert = CustomBertModel(CHECKPOINT_NAME)
bert.to(device)
```

<pre>
CustomBertModel(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(42000, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (2): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (3): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (4): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (5): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (6): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (7): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (8): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (9): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (10): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dr): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=768, out_features=2, bias=True)
)
</pre>

```python
# loss 정의: CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 정의: bert.paramters()와 learning_rate 설정
optimizer = optim.Adam(bert.parameters(), lr=1e-5)
```

학습(train) 함수를 정의합니다. 기존의 PyTorch 모델 학습과 별반 다르지 않습니다.



```python
from tqdm import tqdm  # Progress Bar 출력

def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()
    
    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    counts = 0
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    
    # mini-batch 학습을 시작합니다.
    for idx, (inputs, labels) in enumerate(prograss_bar):
        # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        inputs = {k:v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        # 누적 Gradient를 초기화 합니다.
        optimizer.zero_grad()
        
        # Forward Propagation을 진행하여 결과를 얻습니다.
        output = model(**inputs)
        
        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = loss_fn(output, labels)
        
        # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산합니다.
        loss.backward()
        
        # 계산된 Gradient를 업데이트 합니다.
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
        _, pred = output.max(dim=1)
        
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
        # 합계는 corr 변수에 누적합니다.
        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        
        # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
        running_loss += loss.item() * labels.size(0)
        
        # 프로그레스바에 학습 상황 업데이트
        prograss_bar.set_description(f"training loss: {running_loss/(idx+1):.5f}, training accuracy: {corr / counts:.5f}")
        
    # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됩니다.
    acc = corr / len(data_loader.dataset)
    
    # 평균 손실(loss)과 정확도를 반환합니다.
    # train_loss, train_acc
    return running_loss / len(data_loader.dataset), acc
```

평가(evaluation)를 위한 함수를 정의합니다.



```python
def model_evaluate(model, data_loader, loss_fn, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        
        # 배치별 evaluation을 진행합니다.
        for inputs, labels in data_loader:
            # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(**inputs)
            
            # output의 max(dim=1)은 max probability와 max index를 반환합니다.
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
            _, pred = output.max(dim=1)
            
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
            # 합계는 corr 변수에 누적합니다.
            corr += torch.sum(pred.eq(labels)).item()
            
            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss_fn(output, labels).item() * labels.size(0)
        
        # validation 정확도를 계산합니다.
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출합니다.
        acc = corr / len(data_loader.dataset)
        
        # 결과를 반환합니다.
        # val_loss, val_acc
        return running_loss / len(data_loader.dataset), acc
```


```python
# 최대 Epoch을 지정합니다.
num_epochs = 10

# checkpoint로 저장할 모델의 이름을 정의 합니다.
model_name = 'bert-kor-base'

min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    # Model Training
    # 훈련 손실과 정확도를 반환 받습니다.
    train_loss, train_acc = model_train(bert, train_loader, loss_fn, optimizer, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(bert, test_loader, loss_fn, device)   
    
    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(bert.state_dict(), f'{model_name}.pth')
    
    # Epoch 별 결과를 출력합니다.
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
```

<pre>
training loss: 5.03556, training accuracy: 0.67800: 100% 125/125 [00:20<00:00,  6.23batch/s]
</pre>
<pre>
[INFO] val_loss has been improved from inf to 0.37132. Saving Model!
epoch 01, loss: 0.62945, acc: 0.67800, val_loss: 0.37132, val_accuracy: 0.84800
</pre>
<pre>
training loss: 3.53300, training accuracy: 0.81700: 100% 125/125 [00:20<00:00,  6.19batch/s]
</pre>
<pre>
epoch 02, loss: 0.44163, acc: 0.81700, val_loss: 0.40988, val_accuracy: 0.82600
</pre>
<pre>
training loss: 2.23064, training accuracy: 0.90100: 100% 125/125 [00:20<00:00,  6.19batch/s]
</pre>
<pre>
[INFO] val_loss has been improved from 0.37132 to 0.36758. Saving Model!
epoch 03, loss: 0.27883, acc: 0.90100, val_loss: 0.36758, val_accuracy: 0.86600
</pre>
<pre>
training loss: 1.40625, training accuracy: 0.93100: 100% 125/125 [00:20<00:00,  6.17batch/s]
</pre>
<pre>
epoch 04, loss: 0.17578, acc: 0.93100, val_loss: 0.41405, val_accuracy: 0.86600
</pre>
<pre>
training loss: 1.00858, training accuracy: 0.95900: 100% 125/125 [00:20<00:00,  6.16batch/s]
</pre>
<pre>
epoch 05, loss: 0.12607, acc: 0.95900, val_loss: 0.40552, val_accuracy: 0.86600
</pre>
<pre>
training loss: 0.46512, training accuracy: 0.98100: 100% 125/125 [00:20<00:00,  6.15batch/s]
</pre>
<pre>
epoch 06, loss: 0.05814, acc: 0.98100, val_loss: 0.54145, val_accuracy: 0.86400
</pre>
<pre>
training loss: 0.21964, training accuracy: 0.99200: 100% 125/125 [00:20<00:00,  6.17batch/s]
</pre>
<pre>
epoch 07, loss: 0.02746, acc: 0.99200, val_loss: 0.68765, val_accuracy: 0.85600
</pre>
<pre>
training loss: 0.34378, training accuracy: 0.98700: 100% 125/125 [00:20<00:00,  6.16batch/s]
</pre>
<pre>
epoch 08, loss: 0.04297, acc: 0.98700, val_loss: 0.51284, val_accuracy: 0.87400
</pre>
<pre>
training loss: 0.32060, training accuracy: 0.98900: 100% 125/125 [00:20<00:00,  6.16batch/s]
</pre>
<pre>
epoch 09, loss: 0.04008, acc: 0.98900, val_loss: 0.68871, val_accuracy: 0.84600
</pre>
<pre>
training loss: 0.25765, training accuracy: 0.99100: 100% 125/125 [00:20<00:00,  6.14batch/s]
</pre>
<pre>
epoch 10, loss: 0.03221, acc: 0.99100, val_loss: 0.65872, val_accuracy: 0.86000
</pre>



## 예시 문장을 입력하여 긍/부정 예측하기

먼저, 학습시 저장한 `state_dict`를 로드합니다.

```python
# 저장한 state_dict를 로드 합니다.
bert.load_state_dict(torch.load(f'{model_name}.pth'))
```



학습된 모델을 활용하여 편하게 예측할 수 있도록 `CustomPredictor` 라는 클래스를 정의합니다.

```python
class CustomPredictor():
    def __init__(self, model, tokenizer, labels: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        
    def predict(self, sentence):
        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )
        tokens.to(device)
        prediction = self.model(**tokens)
        prediction = F.softmax(prediction, dim=1)
        output = prediction.argmax(dim=1).item()
        prob, result = prediction.max(dim=1)[0].item(), self.labels[output]
        print(f'[{result}]\n확률은: {prob*100:.3f}% 입니다.')
```



`CustomPredictor` 클래스 인스턴스를 생성합니다. 생성시 학습된 모델, 토크나이저, 라벨 등을 지정합니다.

```python
# Huggingface 토크나이저 생성
tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_NAME)

labels = {
    0: '부정 리뷰 입니다.', 
    1: '긍정 리뷰 입니다.'
}

# CustomPredictor 인스턴스를 생성합니다.
predictor = CustomPredictor(bert, tokenizer, labels)
```



간단하게 문장의 입력을 받으면 예측할 수 있는 함수를 생성하고 실행합니다.

```python
# 사용자 입력에 대하여 예측 후 출력을 낼 수 있는 간단한 함수를 생성합니다.
def predict_sentence(predictor):
    input_sentence = input('문장을 입력해 주세요: ')
    predictor.predict(input_sentence)
```

```python
# 부정 리뷰 입력 예시
predict_sentence(predictor)
```

```
문장을 입력해 주세요: 이 영화는 정말 더럽게 재미없네... 다신 안보련다 비추!
[부정 리뷰 입니다.]
확률은: 93.257% 입니다.
```

```python
# 긍정 리뷰 입력 예시
predict_sentence(predictor)
```

```
문장을 입력해 주세요: 인생 최고의 영화였다. 진짜 배우들의 명연기가 돋보이는 영화
[긍정 리뷰 입니다.]
확률은: 99.676% 입니다.
```

