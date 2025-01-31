---
layout: page
title: "PyTorch의 자동미분(AutoGrad)을 활용하여 정형 데이터 예측기(regression model)구현"
description: "PyTorch의 자동미분(AutoGrad)을 활용하여 정형 데이터 예측기(regression model)에 대해 알아보겠습니다."
headline: "PyTorch의 자동미분(AutoGrad)을 활용하여 정형 데이터 예측기(regression model)에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, pytorch, 파이토치, 경사하강법, gradient descent, 파이토치 입문, 정형데이터, 보스톤 주택가격, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2022-08-06
---

이번 튜토리얼에서는 scikit-learn의 내장 데이터셋인 보스톤 주택 가격 데이터셋을 활용하여 회귀 예측 모델(regression model)을 만들고 예측해 보도록 하겠습니다.

PyTorch의 자동 미분 기능을 활용하여 구현하였으며, 자동미분 기능에 대한 내용은 이전 포스팅에서 공유하였습니다.(아래 링크를 참고해 주세요)

- [경사하강법 구현](https://teddylee777.github.io/scikit-learn/gradient-descent)

- [PyTorch의 자동미분(AutoGrad)기능과 경사하강법(Gradient Descent) 구현](https://teddylee777.github.io/pytorch/pytorch-gradient-descent)

- [경사하강법 기본 개념(YouTube)](https://www.youtube.com/watch?v=GEdLNvPIbiM)


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


## 데이터셋 분할 (x, y)



```python
# feature(x), label(y)로 분할
x = df.drop('target', 1)
y = df['target']

# feature 변수의 개수 지정
NUM_FEATURES = len(x.columns)
```

## 데이터 정규화



- `sklearn.preprocessing 의 `StandardScaler`나 `MinMaxScaler`로 특성(feature) 값을 표준화 혹은 정규화를 진행합니다.



```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled[:5]
```

<pre>
array([[-0.41978194,  0.28482986, -1.2879095 , -0.27259857, -0.14421743,
         0.41367189, -0.12001342,  0.1402136 , -0.98284286, -0.66660821,
        -1.45900038,  0.44105193, -1.0755623 ],
       [-0.41733926, -0.48772236, -0.59338101, -0.27259857, -0.74026221,
         0.19427445,  0.36716642,  0.55715988, -0.8678825 , -0.98732948,
        -0.30309415,  0.44105193, -0.49243937],
       [-0.41734159, -0.48772236, -0.59338101, -0.27259857, -0.74026221,
         1.28271368, -0.26581176,  0.55715988, -0.8678825 , -0.98732948,
        -0.30309415,  0.39642699, -1.2087274 ],
       [-0.41675042, -0.48772236, -1.30687771, -0.27259857, -0.83528384,
         1.01630251, -0.80988851,  1.07773662, -0.75292215, -1.10611514,
         0.1130321 ,  0.41616284, -1.36151682],
       [-0.41248185, -0.48772236, -1.30687771, -0.27259857, -0.83528384,
         1.22857665, -0.51117971,  1.07773662, -0.75292215, -1.10611514,
         0.1130321 ,  0.44105193, -1.02650148]])
</pre>
## PyTorch를 활용하여 회귀(regression) 예측


random 텐서 `w`, 와 `b`를 생성합니다.



`w`의 `Size()`는 `13개`입니다. 이유는 특성(feature) 변수의 개수와 동일해야 합니다.



```python
# random w, b 생성
w = torch.randn(NUM_FEATURES, requires_grad=True, dtype=torch.float64)
b = torch.randn(1, requires_grad=True, dtype=torch.float64)

# w의 Size()는 13, b는 1개 생성
w.shape, b.shape
```

<pre>
(torch.Size([13]), torch.Size([1]))
</pre>
손실함수(Mean Squared Error)를 정의 합니다.



```python
# Mean Squared Error(MSE) 오차 정의
loss_fn = lambda y_true, y_pred: ((y_true - y_pred)**2).mean()
```

`x`, `y`를 tensor로 변환합니다.



```python
x = torch.tensor(x_scaled)
y = torch.tensor(y.values)

x.shape, y.shape
```

<pre>
(torch.Size([506, 13]), torch.Size([506]))
</pre>
단순 선형회귀 생성(simple linear regression)



```python
y_hat = torch.matmul(x, w)
print(y_hat.shape)

# y_hat 10개 출력
y_hat[:10].data.numpy()
```

<pre>
torch.Size([506])
</pre>
<pre>
array([-1.09528567, -2.03626834, -2.50430817, -3.45297498, -3.25141853,
       -3.19429977,  0.2703441 ,  1.01406933,  2.12506136,  1.08349359])
</pre>
## 경사하강법을 활용한 회귀 예측



```python
# 최대 반복 횟수 정의
num_epoch = 20000

# 학습율 (learning_rate)
learning_rate = 5e-4

# random w, b 생성
w = torch.randn(NUM_FEATURES, requires_grad=True, dtype=torch.float64)
b = torch.randn(1, requires_grad=True, dtype=torch.float64)

# loss, w, b 기록하기 위한 list 정의
losses = []

for epoch in range(num_epoch):
    # Affine Function
    y_hat =  torch.matmul(x, w) + b

    # 손실(loss) 계산
    loss = loss_fn(y, y_hat)
    
    # 손실이 20 보다 작으면 break 합니다.
    if loss < 20:
        break

    # w, b의 미분 값인 grad 확인시 다음 미분 계산 값은 None이 return 됩니다.
    # 이러한 현상을 방지하기 위하여 retain_grad()를 loss.backward() 이전에 호출해 줍니다.
    w.retain_grad()
    b.retain_grad()
    
    # 미분 계산
    loss.backward()
    
    # 경사하강법 계산 및 적용
    # w에 learning_rate * (그라디언트 w) 를 차감합니다.
    w = w - learning_rate * w.grad
    # b에 learning_rate * (그라디언트 b) 를 차감합니다.
    b = b - learning_rate * b.grad
    
    # 계산된 loss, w, b를 저장합니다.
    losses.append(loss.item())

    if epoch % 1000 == 0:
        print("{0:05d} loss = {1:.5f}".format(epoch, loss.item()))
    
print("----" * 15)
print("{0:05d} loss = {1:.5f}".format(epoch, loss.item()))
```

<pre>
00000 loss = 579.66085
01000 loss = 93.57605
02000 loss = 32.96278
03000 loss = 24.42413
04000 loss = 23.00189
05000 loss = 22.62038
06000 loss = 22.43333
07000 loss = 22.30946
08000 loss = 22.21997
09000 loss = 22.15347
10000 loss = 22.10328
11000 loss = 22.06487
12000 loss = 22.03509
13000 loss = 22.01169
14000 loss = 21.99307
15000 loss = 21.97806
16000 loss = 21.96581
17000 loss = 21.95572
18000 loss = 21.94731
19000 loss = 21.94024
------------------------------------------------------------
19999 loss = 21.93426
</pre>
## weight 출력



- 음수: 종속변수(주택가격)에 대한 반비례

- 양수: 종속변수(주택가격)에 대한 정비례

- 회귀계수:

  - 계수의 값이 커질 수록 종속변수(주택가격)에 더 크게 영향을 미침을 의미

  - 계수의 값이 0에 가깝다면 종속변수(주택가격)에 영향력이 없음을 의미



```python
pd.DataFrame(list(zip(df.drop('target', 1).columns, w.data.numpy())), columns=['features', 'weights']) \
.sort_values('weights', ignore_index=True)
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
      <th>features</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LSTAT</td>
      <td>-3.732330</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DIS</td>
      <td>-3.036964</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PTRATIO</td>
      <td>-2.033768</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NOX</td>
      <td>-1.952553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TAX</td>
      <td>-1.540389</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CRIM</td>
      <td>-0.878551</td>
    </tr>
    <tr>
      <th>6</th>
      <td>INDUS</td>
      <td>-0.043276</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AGE</td>
      <td>-0.004709</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CHAS</td>
      <td>0.708043</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B</td>
      <td>0.849340</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ZN</td>
      <td>0.995173</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RAD</td>
      <td>2.165836</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RM</td>
      <td>2.723026</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 전체 loss 에 대한 변화량 시각화
plt.figure(figsize=(14, 6))
plt.plot(losses[:500], c='darkviolet', linestyle=':')

plt.title('Losses over epoches', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
```

<img src="data:image/png;base64,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"/>
