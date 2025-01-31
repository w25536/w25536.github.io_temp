---
layout: page
title: "PyTorch 정형데이터를 CustomDataset으로 서브클래싱(SubClassing)한 후 예측 모델 생성 및 학습하기"
description: "PyTorch 정형데이터를 CustomDataset으로 서브클래싱(SubClassing)한 후 예측 모델 생성 및 학습하기에 대해 알아보겠습니다."
headline: "PyTorch 정형데이터를 CustomDataset으로 서브클래싱(SubClassing)한 후 예측 모델 생성 및 학습하기에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, pytorch, pytorch dataset, custom dataset, DataLoader, 데이터로더, 파이토치 데이터셋, 파이토치, 파이토치 입문, 정형데이터, 보스톤 주택가격, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2022-08-08
---


정형데이터셋을 로드하여 PyTorch의 `torch.utils.data.Dataset`을 상속받아 커스텀 데이터셋(CustomDataset)을 서브클래싱(SubClassing)으로 정의하고, 이를 `torch.utils.data.DataLoader`에 주입하여 배치구성을 한 뒤, 모델 학습까지 진행하는 튜토리얼입니다.


튜토리얼에 활용한 데이터셋은 `sklearn`의 내장 데이터셋인 Boston Housing Price 데이터셋을 활용하였습니다.


파이토치 코리아의 [Dataset / DataLoader 튜토리얼](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)을 참고하였습니다. 


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


## 샘플 데이터셋 로드



```python
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_boston
import torch

warnings.filterwarnings('ignore')
```


```python
# sklearn.datasets 내장 데이터셋인 보스톤 주택 가격 데이터셋 로드
data = load_boston()
```

**컬럼 소개**



속성 수 : 13



* **CRIM**: 자치시 별 범죄율

* **ZN**: 25,000 평방 피트를 초과하는 주거용 토지의 비율

* **INDUS**: 비소매(non-retail) 비즈니스 토지 비율

* **CHAS**: 찰스 강과 인접한 경우에 대한 더비 변수 (1= 인접, 0= 인접하지 않음)

* **NOX**: 산화 질소 농도 (10ppm)

* **RM**:주택당 평균 객실 수

* **AGE**: 1940 년 이전에 건축된 자가소유 점유 비율

* **DIS**: 5 개의 보스턴 고용 센터까지의 가중 거리     

* **RAD**: 고속도로 접근성 지수

* **TAX**: 10,000 달러 당 전체 가치 재산 세율

* **PTRATIO**  도시별 학생-교사 비율

* **B**: 인구당 흑인의 비율. 1000(Bk - 0.63)^2, (Bk는 흑인의 비율을 뜻함)

* **LSTAT**: 하위 계층의 비율

* **target**: 자가 주택의 중앙값 (1,000 달러 단위)



```python
# 데이터프레임 생성. 504개의 행. Feature: 13개, target은 예측 변수(주택가격)
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
print(df.shape)
df.head()
```

<pre>
(506, 14)
</pre>
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



```python
# feature 변수의 개수 지정
NUM_FEATURES = len(df.drop('target', 1).columns)
print(f'number of features: {NUM_FEATURES}')
```

<pre>
number of features: 13
</pre>
## 서브클래싱으로 CustomDataset 생성



- SubClassing으로 Dataset을 상속받아 구현하게 되면 DataLoader에 주입하여 배치(batch) 구성을 쉽게 할 수 있습니다.

- 보통 `__init__()` 함수에서 데이터를 set 해주게 되고, 기타 필요한 전처리를 수행합니다. Image Transformation은 `__getitem__(self, idx)`에서 구현하는 경우도 있습니다.

- SubClassing으로 커스텀 Dataset을 구성한다면 `__len__(self)`함수와 `__getitem__(self, idx)`를 구현해야 합니다.

- [참고: 파이토치 튜토리얼(Tutorials > Dataset과 DataLoader)](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)



```python
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, data, target='target', normalize=True):
        super(CustomDataset, self).__init__()
        self.x = data.drop(target, 1)
        
        # 데이터 표준화
        if normalize:
            scaler = StandardScaler()
            self.x = pd.DataFrame(scaler.fit_transform(self.x))
        
        self.y = data['target']
        
        # 텐서 변환
        self.x = torch.tensor(self.x.values).float()
        self.y = torch.tensor(self.y).float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
```


```python
# Custom으로 정의한 데이터셋 생성
dataset = CustomDataset(df, 'target', True)
```

Custom으로 정의한 데이터셋은 `torch.utils.data.DataLoader`에 주입할 수 있습니다.



```python
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, 
                         batch_size=32, 
                         shuffle=True)
```


```python
x, y = next(iter(data_loader))
```


```python
x.shape, y.shape
```

<pre>
(torch.Size([32, 13]), torch.Size([32]))
</pre>
## PyTorch를 활용하여 회귀(regression) 예측



```python
# Device 설정 (cuda:0 혹은 cpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

<pre>
cuda:0
</pre>

```python
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 8)
        # 마지막 출력층의 Neuron은 1개로 설정
        self.output = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
```


```python
# 모델 생성
model = Net(NUM_FEATURES)
# 모델을 device 에 올립니다. (cuda:0 혹은 cpu)
model.to(device)
model
```

<pre>
Net(
  (fc1): Linear(in_features=13, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=8, bias=True)
  (output): Linear(in_features=8, out_features=1, bias=True)
)
</pre>
## 손실함수(Loss Function) / 옵티마이저(Optimzier) 정의



```python
# Mean Squared Error(MSE) 오차 정의
loss_fn = nn.MSELoss()
```


```python
# 옵티마이저 설정: model.paramters()와 learning_rate 설정
optimizer = optim.Adam(model.parameters(), lr=0.005)
```

## 경사하강법을 활용한 회귀 예측



```python
# 최대 반복 횟수 정의
num_epoch = 200

# loss 기록하기 위한 list 정의
losses = []

for epoch in range(num_epoch):
    # loss 초기화
    running_loss = 0
    for x, y in data_loader:
        # x, y 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        x = x.to(device)
        y = y.to(device)
    
        # 그라디언트 초기화 (초기화를 수행하지 않으면 계산된 그라디언트는 누적됩니다.)
        optimizer.zero_grad()

        # output 계산: model의 __call__() 함수 호출
        y_hat =  model(x)

        # 손실(loss) 계산
        loss = loss_fn(y, y_hat)

        # 미분 계산
        loss.backward()

        # 경사하강법 계산 및 적용
        optimizer.step()

        # 배치별 loss 를 누적합산 합니다.
        running_loss += loss.item()
        
    # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
    loss = running_loss / len(data_loader)
    losses.append(loss)

    # 20번의 Epcoh당 출력합니다.
    if epoch % 20 == 0:
        print("{0:05d} loss = {1:.5f}".format(epoch, loss))
    
print("----" * 15)
print("{0:05d} loss = {1:.5f}".format(epoch, loss))
```

<pre>
00000 loss = 553.85693
00020 loss = 86.32502
00040 loss = 85.68998
00060 loss = 84.82237
00080 loss = 85.05138
00100 loss = 84.69042
00120 loss = 85.61478
00140 loss = 85.20934
00160 loss = 85.58127
00180 loss = 84.89186
------------------------------------------------------------
00199 loss = 84.44175
</pre>

```python
# 전체 loss 에 대한 변화량 시각화
plt.figure(figsize=(14, 6))
plt.plot(losses[:100], c='darkviolet', linestyle=':')

plt.title('Losses over epoches', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
```

<img src="data:image/png;base64,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"/>
