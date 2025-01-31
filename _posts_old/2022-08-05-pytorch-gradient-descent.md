---
layout: page
title: "PyTorch의 자동미분(AutoGrad)기능과 경사하강법(Gradient Descent) 구현"
description: "PyTorch의 자동미분(AutoGrad)기능과 경사하강법(Gradient Descent) 구현방법 대해 알아보겠습니다."
headline: "PyTorch의 자동미분(AutoGrad)기능과 경사하강법(Gradient Descent) 구현방법 대해 알아보겠습니다"
categories: pytorch
tags: [python, 파이썬, pytorch, 파이토치, 경사하강법, gradient descent, 파이토치 입문, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
use_math: true
typora-copy-images-to: ../images/2022-08-06
---

이번 포스팅에서는 `PyTorch`의 자동미분(AutoGrad) 기능을 활용하여 경사하강법 알고리즘을 직접 구현해보고 손실(loss) 값과 `weights`, `bias`의 변화량을 시각화해 보겠습니다.


**실습파일** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teddylee777/machine-learning/blob/master/02-PyTorch/03-pytorch%EB%A1%9C-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-%EA%B5%AC%ED%98%84.ipynb)




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


## PyTorch로 경사하강법(Gradient Descent) 구현



기본 개념은 함수의 기울기(경사)를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지 반복시키는 것입니다.



**비용 함수 (Cost Function 혹은 Loss Function)를 최소화**하기 위해 반복해서 파라미터를 업데이트 해 나가는 방식입니다.



경사하강법에 대한 상세한 설명은 아래 링크를 참고해 주시기 바랍니다.



- [경사하강법 구현](https://teddylee777.github.io/scikit-learn/gradient-descent)

- [경사하강법 기본 개념(YouTube)](https://www.youtube.com/watch?v=GEdLNvPIbiM)



```python
# 모듈 import 
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
```

## 샘플 데이터셋 생성



- `y = 0.3x + 0.5`의 선형회귀 식을 추종하는 샘플 데이터셋을 생성합니다.

- 경사하강법 알고리즘으로 `w=0.3`, `b=0.5`를 추종하는 결과를 도출하도록 하겠습니다.



```python
def make_linear(w=0.5, b=0.8, size=50, noise=1.0):
    x = np.random.rand(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='r', label=f'y = {w}x + {b}', linestyle=':', alpha=0.3)
    plt.scatter(x, yy, color='black', label='data', marker='.')
    plt.legend(fontsize=15)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy

x, y = make_linear(w=0.3, b=0.5, size=100, noise=0.01)
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlkAAAGbCAYAAAD3MIVlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAABAbUlEQVR4nO3deZzcVZ3v/9fp6u5AQCAkBiUhISzSLC1imkCzDB0jWYRmkdEHzPURlE0ZEPXyy1y9g+KSa5zxOjoqahgGSHxcV64jCasY0qDQOOnoZdqEgBAwJIBIElkk6U51nd8f1d2pqt6q16rqej0fjzzM93yXOm0F8uac8/2cEGNEkiRJI6ui0B2QJEkajwxZkiRJo8CQJUmSNAoMWZIkSaPAkCVJkjQKKgvdgVxTpkyJhx9+eKG7IUmSNKD169e/EmN8a2/nii5kHX744bS0tBS6G5IkSQMKIfyxr3NOF0qSJI0CQ5YkSdIoMGRJkiSNAkOWJEnSKDBkSZIkjQJDliRJ0ijIq4RDCGEh8K9AArglxviVnPMzgBXAQZ3XfDrGeE/nuc8AlwMdwHUxxvuH2+nXXnuNl19+mT179gz3UdKgVVVVMXXqVA444IBCd0WSVMQGDFkhhARwE3A2sBVYF0JYFWPcmHHZDcBPYozfDSEcB9wDHN75+4uB44FDgV+GEN4RY+wYaodfe+01/vSnPzFt2jT23XdfQghDfZQ0aDFGdu3axbZt2wAMWpKkPuUzXTgHeDrGuDnG2A78CDg/55oIdP1tcyDwQufvzwd+FGNsizE+Czzd+bwhe/nll5k2bRoTJ040YGnMhRCYOHEi06ZN4+WXXy50dyRJRSyfkDUNeD7jeGtnW6bPAx8KIWwlPYr18UHcSwjhqhBCSwih5c9//nO/ndmzZw/77rtvHt2WRs++++7rdLUkqV8jtfD9EuD2GON04H3A90MIeT87xnhzjLEuxlj31rf2uv1PFkewVGj+GZQkDSSfhe/bgMMyjqd3tmW6HFgIEGNsDiHsA0zJ815JkqRxJ5/RpnXA0SGEWSGEatIL2VflXLMFmAcQQjgW2Af4c+d1F4cQJoQQZgFHA/85Up2XJEkqVgOGrBhjErgWuB94gvRbhBtCCF8MIZzXedn1wJUhhMeBHwIfjmkbgJ8AG4H7gGuG82ahRsbGjRuZN28eEydO5NBDD+Vzn/scHR39fy0bNmxg4cKFHHrooUyYMIEZM2ZwxRVX8OKLL45Rr/Nz5513Ultbyz777MNxxx3Hj3/84wHvuf322wkh9Pj1ve99bwx6LEkar/Kqk9VZ8+qenLbPZfx+I3B6H/f+L+B/DaOPGkE7d+7kve99L8cddxx33nknzzzzDNdffz2pVIqlS5f2ed+rr77KrFmzWLx4MYceeijPPvssX/jCF1i/fj3r1q2jsjKvP0qj6te//jUXXXQRf//3f883v/lN7rnnHi655BImTZrE/PnzB7z/wQcfzHqp4ogjjhjN7kqSxrsYY1H9mj17duzPxo0b+z2v/n35y1+OBx10UHz11Ve72/7pn/4p7rvvvllt+fjFL34Rgbh+/fqR7macOXNmvO222wZ1z/z58+PcuXOz2hYtWhRPP/30fu+77bbbIhBff/31QX2efxYlqTAeffTR+OUvfzk++uijvV/w8ssxplJj0hegJfaRadxWZ4zdc889VFRU8Oyzz2a1P/vss1RUVHDnnXeO6uffe++9LFiwIKuI5sUXX8yuXbt46KGHBvWsyZMnA9De3g7AY489RmVlJbfeemv3Na+++iqHHXYY/+2//bcR6H3f2traWLt2LR/84Aez2i+++GKam5t59dVXR/XzJUmjp7m5mWXLltHc3ExzczPz5s3js5/9LPPmzaO5uTn74vXr4bHH4OmnC9PZDIasMbZgwQIOPfRQVqxYkdV+++23M3XqVM4555w+7+3o6CCZTPb7K5VK9fv5mzZtoqamJqttxowZTJw4kU2bNg3Y/1QqRXt7O08++SSf/vSnOfnkk5kzJ11f9tRTT2XJkiV86lOfYsuWLQBcd911pFIpvv3tbw/47OF45pln2LNnT4+f7dhjjyWVSvHUU08N+IwjjzySyspKjjnmGJYvXz5aXZUkDUJuqFq5ciXt7e10dHTQ3t5OU1NT+sKutcXHHAOJBI+9+GJ3MCuU8ROyHn0Unu+se5pKpY+3bk0fd3Skj1/oLES/Z0/6uGvRdnt7+vhPf0oft7Wlj7sqeu/alT7uKpT65pvp4+3bB93NRCLBhz/8YVasWEF6lDE9ZbtixQo+9KEP9bu2ad68eVRVVfX767LLLuv383fu3MlBBx3Uo33SpEns3LlzwP6/733vY8KECdTU1LBjxw7uuusuKir2/jH6whe+wMyZM7nsssu48847WblyJbfccguTJk3q97m5YRHSgS6zrev/r75+LqDHz9b1uf39bG9/+9v50pe+xPe//31Wr17Nqaeeysc+9jG+/vWv99tnSdLwZI5Q9aWpqSkrVAFUV1eTSCSorq6moaEB7rsP7rkn/ff5/vvTPGkS71m4sO/RrjFS+NXKZeiyyy7jy1/+Mk1NTcydO5e1a9fyxz/+kY985CP93rd8+XJef/31fq+ZMmXKSHa1h29961vs2LGDP/zhDyxdupRFixbxyCOPsM8++wDpP/grV65kzpw5/PrXv+aKK65g0aJFAz63qqqqR9vll1/O5Zdf3n1822238eEPf3jEfpYuCxYsYMGCBd3HixYtYvfu3SxdupRPfOITWSFSkjQyukao2tvbqa6uZs2aNUA6VDU0NFBfXw9AQ0MD1dXV3dctXryYxYsXZ1/XNfDRWSg6N5g1NTV1P28sjZ+Qddppe39fUZF9nEhkH1dVZR9XV2cfT5iQfbzvvtnHEydmHw/SEUccQUNDA7fddhtz587ltttuY86cORx//PH93nfUUUf1O5oDDBgIJk2a1Ov6pJ07dw442gRw9NFHA3DKKadw5plnMmvWLH7wgx9kjaCdeOKJHHfccTz++OP8/d///YDPBFi3bl3W8XnnncdVV13Fueee2902a9asPu/v6nvuz9Y1gpXPz5bpb//2b/nJT37Cc88951uGkjQKcoPQypUrWbFiBe3t7SQSCS677DIWL15MfX09a9asyQ5VMVK/c2f672+A+vrugAU9g1lDQ0NBfsbxE7JKzBVXXMGVV17JsmXL+NnPfsbXvva1Ae+ZN2/egIvTL730Um6//fY+z9fU1PRYe/X888/z5ptv9ljPNJCZM2dy8MEHs3nz5qz2b3zjG2zatIljjz2W6667joceemjA8FdXV5d1XF1dzeGHH96jvS9HHnkkVVVVbNq0ibPOOqu7fdOmTVRUVPCOd7wjz58qrWvbHLfPkaTRkRuEgO7Q1dHRwfLly1mxYgVr1qyhvr5+70jUSy+lB1M6OtLLgGbPzgpYQO/BrAAMWQXy/ve/n2uuuYaLL76YVCrFxRdfPOA9IzFduGjRIr761a/y+uuv85a3vAWAH//4x+y7775Z4SQfTz75JNu3b88aYXryySf5x3/8R5YuXcrChQuZPXs2X//617n++usH9ezBmjBhAnPnzuWnP/0pH/3oR7vbf/zjH1NfX8+BBx44qOfdcccdTJkyhZkzZ450VyVJ9AxCACtWrGD37t3dJRDa2tr2TvXFCA89BF1/D773vemZpn6eX6hw1a2v2g6F+lVOdbKuueaaCMRLLrlkzD5zx44d8W1ve1t873vfGx944IG4fPnyuN9++8V//Md/zLruyCOPjJdddln38fXXXx//x//4H/FnP/tZfPDBB+NNN90UZ86cGY888sj4xhtvxBhjTCaT8ZRTTomnnXZa7OjoiDHGuGzZsrjPPvvEJ554YlD9HEqdrF/96lcxkUjET3ziE3Ht2rVxyZIlMYQQ77///u5rnnvuuZhIJOKKFSu6297//vfHr3zlK/Gee+6Jq1evjh/60IciEL/5zW/2+3nj6c+iJBWDRx99NF5wwQUR6P61fPnyGP/61xiffjrGVavSv/7yl0J3tRvWySpOF1xwAcCAbwSOpEmTJrFmzRo6OjpobGzkxhtv5FOf+hRf+MIXsq5LJpNZW+3U1dXxq1/9issvv5xzzjmHb37zm1x00UU89thj7LfffgD88z//M62trdx+++3d04NLlizhXe96F5deeumAW/cM1xlnnMEdd9zBL3/5SxYsWMCqVav4wQ9+kFXtPcZIR0dHVqmLY445hltvvZWLLrqID3zgA2zcuJGVK1fy8Y9/fFT7K0nKVl9fz5w5c7r/DqkIgUm//jWsWQO7d8Nxx0FjIwxydqJQQhxgIfVYq6uriy0tLX2ef+KJJzj22GPHsEej5x/+4R/4yU9+wubNm32DrQSNpz+LklQsMt86vCCRYOnSpek1w6edBp1FsItJCGF9jLHXBcSuySqAJ598ko0bN/Ld736XG2+80YAlSVKn+vp6fvuFL9Da2sqsv/1bakKAc85JL3YvMYasAvjoRz/Kb37zG8477zyuu+66QndHkqTi8Npr8NBD1NTUpEevzjoLMraBKzWGrALo3gJAkiSlPfII7Nix93jRIuhnF5RSUNq9lyRJpW3HjnTA6nL66XDwwYXrzwgyZEmSpMLYtAn+8Ie9xwsXpndlGScMWZIkaWz95S/wq1/tPT7qKBiHb2sbsiRJ0tjZuhV+97u9xwNUbi9lhixJkjRimpube98zcNcu+OUvsy9ubBzbzo0xQ5YkSRoRmYVEq6uruzd35umn4Ykn9l549tmwzz6F6+gYKb3KXuPY73//e0IIgyrxcPPNN/Pzn/981PokSVK+mpqaaG9vp6Ojg/b2dh5eswZWr84OWI2NZRGwwJBV8gxZkqRi0dDQQHV1NYlEgndWVnL+hAl7T5599rifHszldKEkSRoR9fX1rHngAbYuX05tbW26ajuUXbjq4khWAX3nO9/hsMMOY7/99qOxsZEXX3wx6/zXvvY1Tj75ZA488EAOOeQQGhsbefrpp7vPNzQ0sH79elasWEEIgRACt99+OwArV67kjDPO4OCDD2bSpEnMnTuX/jbeliRp2H77W+p37OADH/hAOmDV15dtwAJHsgrmzjvv5JprruFjH/sYF1xwAQ899BCXXXZZ1jVbt27l2muvZebMmbz22mt873vf47TTTuMPf/gDBx54IN/5zne46KKLOOKII/jsZz8LwJFHHgnAc889x+LFiznyyCNpb2/nhz/8IWeeeSYbNmzgiCOOGPOfV5I0jsUIzc2wffvetnPPhRAK16ciEGKMhe5Dlrq6utjfiMsTTzzBsSNYsKzPV01H2Zw5c5g8eTL33ntvd9uVV17JLbfcwtq1a2loaMi6vmsR4dSpU7nppptYvHgxAHV1dZxwwgndI1i9SaVSpFIpTjjhBP7u7/6Oz33uc6PxI5Wdkf6zKEkl6ckn4amn9h4ffzyU0X/MhxDWxxjrejtX1tOFXa+afvazn2XevHk0NzePyecmk0l++9vfcv7552e1v//97886fuyxxzj77LOZPHkylZWVTJw4kTfeeIOnMv8w9+GJJ57gwgsv5JBDDiGRSFBVVcWTTz6Z172SJA2kubmZWz/+cTatWrW38ZxzyipgDaSsQ1buq6aDKZ0wHK+88godHR1MnTo1qz3zeMuWLcyfP58YI8uXL+eRRx5h3bp1TJ06ld27d/f7/Ndff5358+fz/PPP8y//8i/86le/Yt26dZx44okD3itJ0kB+98Mf8rWGBlZ95zvccMMN/H7r1vTaq4qyjhU9lPWarK5XTbuKpuVO0Y2WKVOmkEgkePnll7PaM4/vu+8+3nzzTe688072228/ID0CtmPHjgGf39zczNatW3nggQf2vtkBvPrqqyP0E0iSStWwl8msXs3T995LMpmkI5Xi7mSS2X/5CyeMfFdLXlmHrPr6etasWTPma7IqKys56aSTuPPOO/nYxz7W3f6zn/2s+/e7du2ioqKCysq9X9FPfvITkslk1rOqq6t7jE7t2rULgAkZ9UkeffRRnnvuOWbPnj2iP4skqXT0WZE9H9u2wW9/C0BtbS2VlZXc1dExpoMUpaasQxakg9ZYLnjv8j//5//k/e9/P1dffTUXXnghDz30EPfdd1/3+fe85z10dHTwkY98hMsvv5wNGzbwv//3/+aggw7Kek5NTQ33338/999/P5MnT2bWrFmceuqp7L///lx55ZX8wz/8A1u3buXzn/8806ZNG+OfUpJUTHpbJpPX34GrV2cd1nz841x/xhnMLsCLY6XEydMCufDCC/nWt77F6tWrueCCC/jd737Hv//7v3efr62t5fbbb+c3v/kN5557Lj/4wQ/46U9/yoEHHpj1nBtuuIFjjz2WD37wg5x88smsXr2aQw45hJ/+9Ke89NJLnH/++XzjG9/ge9/7HkcdddRY/5iSpCKSWZE9rxGoHTt6BKyubXHq6+v5zGc+Y8DqR9mXcJCGyj+LkkpR3muycsPV6afDwQePbudKUH8lHMp+ulCSpHIy4DKZ9na4//7stjKu2j4chixJkpSWM3rVUlHBA//1XzRMmZIVzApVyLvUGLIkSSp3ySRk7EAC0DxlSq9vIg7rDcUy48J3SZLK2erV2QHruOOgsbHPgt2FKuRdikpyJCvGSCjzTSdVWMX2wogkDVoqBXffnd2Wsfaqr4LdhSrkXYpKLmRVVVWxa9cuJk6cWOiuqIzt2rWLqqqqQndDkobmoYfgtdf2Hk+eDKedlnVJfwW7FyxYwAsvvMDll1/uVGE/Si5kTZ06lW3btjFt2jT23XdfR7Q0pmKM7Nq1i23btnHIIYcUujuSNDgxwl13Zbedey708Xdp7puIzc3NNDQ00N7eDsDjjz9ObW2tQasPJReyDjjgAABeeOEF9uzZU+DeqBxVVVVxyCGHdP9ZlKSS0NwMr7yy93jiRJg3b1CPaGpqyvq7d1BV48tQyYUsSAct/4KTJClPuYVF3/c+SCQG/ZiGhgaqqqq6R7Jck9W/kgxZkiQpD1u2wOOPZ7cNo7BofX09TU1NrFy5EoDFixc7itUPQ5YkSeNEVpHQzKlBgAULoLp62J8xYMV4dTNkSZI0DnQVCd2/rY31lZUsXbqUmpqa9Em3xSkIQ5YkSeNAU1MT89va6EilSCaTtLa2UnP11bD//oXuWtmy4rskSaVuxw4urKyksrKSREUFlZWVTDdgFZwjWZIkFdiwNlzufHOwpqaGpUuXsuaVV3j3BRe4bqoIGLIkSSqgIW+4/Ne/woMPZjXVLFlCzSj1U4NnyJIkqYB623B5wJCVW/dqxgw48cTR66SGJK+QFUJYCPwrkABuiTF+Jef814G5nYcTgakxxoM6z3UArZ3ntsQYzxuBfkuSNC4MasPl9na4//7sNt8cLFoDhqwQQgK4CTgb2AqsCyGsijFu7LomxvipjOs/DpyU8YhdMcZ3jViPJUkaR/rbiDlL7uhVL5s6q7jkM5I1B3g6xrgZIITwI+B8YGMf118C3Dgy3ZMkafzrt8BnKgV3353d5uhVScgnZE0Dns843gqc0tuFIYSZwCwgcyXePiGEFiAJfCXG+PNe7rsKuApgxowZeXVckqRx7+GH4dVX9x5XVcHChYXrjwZlpBe+XwzcEWPsyGibGWPcFkI4AngwhNAaY3wm86YY483AzQB1dXVxhPskSVJpiRHuuiu77ZxzoGLg8pbDKgehEZVPyNoGHJZxPL2zrTcXA9dkNsQYt3X+7+YQQhPp9VrP9LxVkiTx+9/Ds8/uPZ4wAebPz+vWIZeD0KjIp+L7OuDoEMKsEEI16SC1KveiEEINMAlozmibFEKY0Pn7KcDp9L2WS5Kk8rZ6dXbAWrgw74AFvZeDUOEMOJIVY0yGEK4F7iddwuHWGOOGEMIXgZYYY1fguhj4UYwxc7rvWGB5CCFFOtB9JfOtREmSBGzeDBs2ZLcNYXH7oMpBaNSF7ExUeHV1dbGlpaXQ3ZAkaWzklmZoaIC3vGXIj3NN1tgKIayPMdb1ds6K75IkFcILL8D69dltI1Caod9yEBpThixJksZa7ujVySfD295WmL5o1BiyJEkaK6+8As3N2W0WFh23DFmSJI2BTV/9Kq2trdTW1lJTU5Pe0NkC3OOaIUuSpNH0yitsuu02brjhBpLJJJWVlVzf1ES9AWvcM2RJkjRaOtdetba2kkwmaU2l+GNHB7ObmlycXgYMWZIkjbQ33oC1a7sPa2tr+cWECdavKjOGLEmSBmHAOlS5bw4efTQ1jY2sOeMM61eVGUOWJEl5uvnmm7nmmmtIpVJMmDAhe2/APXvgvvuyb8h4c9D6VeXHkCVJUh6am5u59tprSSaTALS1tdHUtbbqwQfhr3/de/H06XDSSQXqqYqFIUuSpDw0NTXR0dHRfVxRUUHDWWf1nB485xyoqBjj3qkYGbIkScpDQ0MDEyZMoK2tjUQiwR3XXUf99u17L5gwAebPL1wHVXQMWZKkspfPpsr19fWsWbOGpqYmLqysTBcU7bJgAVRXj1FvVSoMWZKkstbc3My8efO6yytkLWbPUf/669SfcEJ2o9viqA9OGkuSylpTUxPt7e10dHSwe/duVq5c2fuFq1dDW9ve43nzDFjqlyFLklTWGhoaqKxMT+zEGLn11ltpztzE+ZFHei5ub2yEiROzmpqbm1m2bFn2vSprThdKkspafX09H/nIR1i+fDkxRjo6OvaWZsgNV2edBQcc0OMZg5lyVPlwJEuSVPYWL17MPvvsQyKRoLq6moWzZvU+etVLwILsKcf29naamppGv9Mqeo5kSZLKXo83B/fbb+/JU06BqVP7vb+hoYHq6mr3JlQWQ5YkSUD94YcP+c3BzJDm3oTqYsiSJCl3avAd74BjjhnUI9ybULkMWZKk8vXGG7B2bffhpk2b+I9kkoYpUzAuabgMWZKk8pQzerXxmWeou/FG3xDUiPHtQklSeUkme31z8M5du3xDUCPKkSxJUvnIDVfQvbjdNwQ10gxZkqTxL0a4667stnPOgYq9Ezq+IaiRZsiSJBWt5ubm4YeefkavcvmGoEaSIUuSVJRGZKua3IC1cCFUVY1cJ6V+uPBdklSUhrVVzerVvW+LY8DSGHIkS5JUlIa8ED03XM2dC/vvP+L9kwZiyJIkFVRf664GvRC9pQVefDG7Lc9tcaTRYMiSJBXMQOuu8l6Injt6ddJJMH36CPdWGhzXZEmSCmZY664AXnih97VXBiwVAUeyJEkFM6wCoLnhavr09AjWAEakLISUB0OWJKlghlQA9NVX4eGHuw8Hs6nziJSFkPJkyJIkFdSgCoDmjF498dRTzB7Eps69TU8asjRaXJMlSSp+u3f3uvbq5+3tg1rT1TU9mUgk3J9Qo86RLElScRvBTZ3dn1BjKcQYC92HLHV1dbGlpaXQ3ZAkFVoyCffem93WS90rF7KrkEII62OMdb2dcyRLklR83NRZ44AhS5JUPGKEu+7Kbnvf+yCRKEx/pGEwZEmSisMvfwm7dmW3uS2OSpghS5JUeLnTg+99L+y7b2H6Io0QQ5YkqXDWrYOXXspuy2P0qrm5mZUrVwKwePFi12SpKBmyJEmFkTt6deqp8Na3Dnhbc3Mzc+fOpa2tDYBbb73VoqIqShYjlSSNiObmZpYtW0Zzc3P/Fz7xRO+bOucRsGBv1fYue/bsGfzG0tIYcCRLkjRsee8JmBuu3vEOOOaYQX1WVwHSrpGsqqoqK7erKBmyJEnDNuCegJs3w4YN2TcN8c3B+vp61q5d65osFT1DliRp2Prd3iZ39OrQQ2H27Lye21c1dwuQqhQYsiRJw9brnoAvvggZ26Rt2rSJ/0gmaZgyhXziUd5TkFKRyitkhRAWAv8KJIBbYoxfyTn/dWBu5+FEYGqM8aDOc5cCN3SeWxpjXDEC/ZYkFZms0aWc0asNzz3HyTfeOKjANOAUpFTkBny7MISQAG4CFgHHAZeEEI7LvCbG+KkY47tijO8CvgX8rPPeg4EbgVOAOcCNIYRJI/oTSJKKx2uv9frm4Ko33ugRmAbSNQWZSCR6TkFKJSCfkaw5wNMxxs0AIYQfAecDG/u4/hLSwQpgAfBAjHFH570PAAuBHw6n05KkItTPps79rtnqQ69TkFIJySdkTQOezzjeSnpkqocQwkxgFvBgP/dO6+W+q4CrAGbMmJFHlyRJRWPPHrjvvuy2nDcHhxqYXOCuUjbSC98vBu6IMXYM5qYY483AzQB1dXVxhPskSRot/YxeQc+3Aw1MKif5hKxtwGEZx9M723pzMXBNzr0NOfc25d89SVJRihHuuiu7bdEiqNz714pvB6rc5bOtzjrg6BDCrBBCNekgtSr3ohBCDTAJyNxP4X5gfghhUueC9/mdbZKkUrV6dc+A1diYFbCg97cDpXIy4EhWjDEZQriWdDhKALfGGDeEEL4ItMQYuwLXxcCPYowx494dIYQvkQ5qAF/sWgQvSSpBudOD73kP7Ldfr5cOZbG7NJ6EjExUFOrq6mJLRvE6SdLo6auieg8DrL0a9vOlEhVCWB9jrOvtnBXfJalMDXlT51NOgalT8/oMF7urnOWzJkuSNA4NuGbq4Yd7LSyab8CSyp0jWZJUpga1qfOxx8JRR+X9bKcJJUOWJI0rgwk3mQVCJ0+eTFNTExOfeYYTDzww+8I81l7l9sHSDZIhS5LGjaGEm67z8+bNY35bG+srK1m6dCk1NTUwYwaceOKg++HGzlKaIUuSSlzX6NWWLVuGFG7WrV7N/LY2OlIpSCZpbW2lZsmSIffH0g1SmiFLkkpY5uhVIpGgsrMgaN7hZvVq5k+axMOVlZBM8lplJdOvvnpYfXJjZynNkCVJJSxzag7gyiuvZMaMGQOHm9deg4ceAqCmpoalS5fyH8nkiIUiSzdIhixJKmm5U3OLFy8eONzkvjlYXU3NkiV8Js/P9M1BKT+GLEkqYYOamtuzB+67L7st583BgQKUbw5K+TNkSVKJy2tqLo9tcfIJUL45KOXPiu+SNJ7F2DNgLVrUa+2rASvAs3d6MpFI+OagNABHsiRpvBrkps75lF7wzUEpfyHGWOg+ZKmrq4stLS2F7oYklbbcgDVvHkycOOBtLmqXBieEsD7GWNfbOUeyJGk8GeToVS5LL0gjxzVZkjRe5Aas006jecoUli1bRnNzc2H6JJUxR7IkqdTdd1+6PEOmxsZ+3xZ0WlAafYYsSSpluaNXxx8PRxwBpN8WbGtrI5VK0dbW1l1uwVpX0thwulCSStH69T0DVmNjd8ACmDx5MqlUCoBUKsXkyZOB/Eo1SBo+R7IkqdR0hqtNmzbR2tpKzd/8DbVXXNF9umsqcMuWLVRUVJBKpaioqGD79u1AfqUaJA2fIUuSSsWWLfD440A6YN1www38vKOD6jvuYM3xx/eYCqysrKSyspKOjo6sMGWtK2lsGLIkqRTkTA22trby846OHtvbZE4FAlx55ZXMmDGjR5iyVIM0+gxZklTMXnsNHnoou62xkelTplB9xx09pvxypwIXL15smJIKxJAlSWNk0GUT+iks2teUn1OBUvFwWx1JGgODKpuwZ0+69lWmc8+FEEa/o5IGpb9tdSzhIEljIO+yCatX9wxYjY0GLKkEOV0oSWNgwLIJMcJdd2W3LVoElf5rWipV/tMrSWOg37VSw9zUWVJxMmRJ0hjptWxCbsCaOxf233/sOiVp1BiyJGkUDPgmoaNX0rhnyJKkETbgm4S5AWv2bDj00LHtpKRRZ8iSpBHW25uE9fX18PDD8Oqr2Rc7eiWNW4YsSRphvb5JmDt6NWMGnHhiQfonaWwYsiRphGW+SbjwiCM46ZVXsi9w9EoqC4YsSRoF9fX11OeGq6oqWLiwMB2SNOYMWZI00l54Adavz25z9EoqO4YsSerHSG7qPOKfJamoGbIkqQ+D2tT59dchdz/CQYxeDeqzJJUEN4iWpD4MalPnYQSsQX2WpJJhyJKkPnSVYkgkEr1v6rxnT4/pwebJk1n2+9/T3Nw8sp8lqeQ4XShJfRjsps7NU6YMecqv38+SVJIMWZLUjx6bOscId92VfdHChVBVxcqrr2b37t3EGLMrvQ/1sySVNEOWJOUrZ/Rq06ZN/EcyScOUKQDcdtttxBgBSCQSTvlJZc6QJUn5yAlY/zlhAg033tg9NXjppZeSTCYBCCFw2WWXOSollTlDliT1p4+6V2uWLct6GxDI2q9w8eLFY9xRScXGkCVJfckNWLNnw6GHAj03gV68eDGLFy924bqkboYsScq1Zg28+WZ2W07dq77eBjRcSepiyJKkTLmjVzNmwIkn9nqpbwNK6o8hS5IAfvtb2LYtu62Xqu3uLygpX4YsScodvaqshEWLug+7gtXkyZP55Cc/6f6CkvKSV8gKISwE/hVIALfEGL/SyzUfBD4PRODxGOPfdbZ3AK2dl22JMZ43Av2WpOHbsgUefzy7LWf0KnPj5hACqVSKVCo1pGKjksrLgCErhJAAbgLOBrYC60IIq2KMGzOuORr4DHB6jHFnCGFqxiN2xRjfNbLdlqRh6qM0Q67MjZsrKipIJBKEENxfUNKA8hnJmgM8HWPcDBBC+BFwPrAx45orgZtijDsBYowvj3RHJWlEvPYaPPRQdlsv4apLbqmGb3zjG2zfvt01WZIGlE/ImgY8n3G8FTgl55p3AIQQHiE9pfj5GON9nef2CSG0AEngKzHGn+d+QAjhKuAqgBkzZgym/5KUvzxHrzK5cbOkoRqphe+VwNFAAzAdeDiEUBtj/AswM8a4LYRwBPBgCKE1xvhM5s0xxpuBmwHq6uriCPVJktL27IH77stuO/dcCCGv2y3VIGko8glZ24DDMo6nd7Zl2gr8Jsa4B3g2hPAU6dC1Lsa4DSDGuDmE0AScBDyDJI2FIYxeSdJIyCdkrQOODiHMIh2uLgb+LueanwOXALeFEKaQnj7cHEKYBLwZY2zrbD8d+OeR6rwk5dat6j4+6yzqt2/PvnjBAqiuLkxHJZWdAUNWjDEZQrgWuJ/0eqtbY4wbQghfBFpijKs6z80PIWwEOoAlMcbtIYTTgOUhhBRQQXpN1sY+PkqSBiWzvELXovRPfvKTzG9rY31lJUuXLqWmpiZ9saNXksZYXmuyYoz3APfktH0u4/cR+O+dvzKveRSoHX43JZWbfCqrZ5ZXaG9v5//+3//L/LY2OlIpSCZpbW2l5oorYNKkMe69JFnxXVIRyh2h6quyemZ5hQsSCa446ihuaWqCZJLKykqmX321AUtSwRiyJBWd3BGqviqrd5VX2Prd71JbW0tNTQ2HH344D2zfTt355/tGoKSCMmRJKjq5BUD7rKy+Zg31b74JH/hAd1PNkiXUjE03JalfhixJRSevAqC5pRlmzoR3vnNsOihJeTBkSSpKfRYA3bQJ/vCH7DbfHJRUhAxZkkpH7ujVwQfD6acXpi+SNABDlqTi99JLsG5ddpujV5KKnCFLUnFzWxxJJcqQJak4vf46NDVltxmuJJUQQ5ak4uPolaRxwJAlqXgkk3Dvvdlt554LIRSmP5I0DIYsScXB0StJ44whS1Lh5QashQuhqqowfZGkEWLIklQ4jl5JGscMWZIKIzdgzZ0L++9fmL5I0igwZEkaW45eSSoThixJYyc3YJ18MrztbYXpiySNMkOWpNH36KOwfXt2m6NXksY5Q5ak0ZU7ejVjBpx4YmH6IkljyJAlaXQ8/TQ88UR2m6NXksqIIUvSiNv01a/S2tpKbW0tNTU1UFEB55xT6G5J0pgyZEkaOX/6E5tWruSGG24gmUxSWVnJ9U1N1NfXD3hrc3MzTU1NNDQ05HW9JBU7Q5akkdG59qq1tZVkMklHKsVdHR3MziNkNTc3M2/ePNrb26murmbNmjUGLUklr6LQHZBU4v7616zF7bW1tfxiwgTuTSSorq6moaFhwEc0NTXR3t5OR0cH7e3tNDU1jV5/JWmMOJIlaeh6KSxas2QJa844Y1BTfw0NDVRXV3ePZOUTzCSp2IUYY6H7kKWuri62tLQUuhuS+pNMwr33Zredey6EMORHuiZLUikKIayPMdb1ds6RLEmDM8xtcfoKU/X19YYrSeOKIUtS/nID1oIFUF2d9+0ucJdUTlz4Lmlgq1f3DFiNjYMKWOACd0nlxZEsSf3LDVdnngkHHTSkR7nAXVI5MWRJ6t0w1171pr6+njVr1rjAXVJZMGRJ6ik3YL3znTBz5og82gXuksqFIUvSXuvWwUsvZbe5qbMkDYkhS1Ja7ujVrFlwwgmF6YskjQOGLKncPfUUPPlkdpujV5I0bIYsqZzljl4dfDCcfnph+iJJ44whSypHL74IudtX5TF65dY3kpQ/Q5ZUboZYmsFq7ZI0OIYsaRzLGnl65zvhwQezL+glXPU1WtVbtXZDliT1zZAljVOZI08XJBIsXbqUmpqavRf0EbD6Gq2yWrskDY4hSxqnmpqaSLa1sSiVIhkjra2t6ZB17rkQQp/39DVaZbV2SRocQ5Y0Tl1YWcn6ykqSySSVlZXU1tYOuPZqoNEqq7VLUv5CjLHQfchSV1cXW3LfepLKyIi8wde5uH3Tpk20trZy2JVXcuqZZ47d50tSmQghrI8x1vV6zpAlFY9hv8E3Cps6S5L61l/IqhjrzkjqW29rovKWG7DOOsuAJUkF5JosqYgM6Q0+R68kqSgZsqQiMug3+HID1kknwfTpo9dBSVLeDFlSkcnrDb516+Cll7LbckavXMAuSYVlyJJKTe7o1RFHwPHHZzW5BY4kFZ4L36VSsXVrz4DV2NgjYMEwF9BLkkZEXiErhLAwhPBkCOHpEMKn+7jmgyGEjSGEDSGEH2S0XxpC+EPnr0tHquNSWVm9Gn73u73H++2XNT3Y3NzMsmXLaG5uBvYuoE8kEm6BI0kFMuB0YQghAdwEnA1sBdaFEFbFGDdmXHM08Bng9BjjzhDC1M72g4EbgTogAus779058j+KNA795S/wq19lt/Wy9qq3qUG3wJGkwspnTdYc4OkY42aAEMKPgPOBjRnXXAnc1BWeYowvd7YvAB6IMe7ovPcBYCHww5HpvjSO5Vmaoa/9Bt0CR5IKK5+QNQ14PuN4K3BKzjXvAAghPAIkgM/HGO/r495puR8QQrgKuApgxowZ+fZdGp/a2uAXv8hu66fu1ZBqa0mSRt1IvV1YCRwNNADTgYdDCLX53hxjvBm4GdLb6oxQn6TSM4TCok4NSlJxyidkbQMOyzie3tmWaSvwmxjjHuDZEMJTpEPXNtLBK/PepqF2Vhq3Uim4++7stnPPhRDyut2pQUkqPvm8XbgOODqEMCuEUA1cDKzKuebndIapEMIU0tOHm4H7gfkhhEkhhEnA/M42SV1Wr+4ZsBob8w5YkqTiNOBIVowxGUK4lnQ4SgC3xhg3hBC+CLTEGFexN0xtBDqAJTHG7QAhhC+RDmoAX+xaBC+JntODCxZAdXVh+iJJGlEhxuJaAlVXVxdbWloK3Q1pdN19d3qKMJObOktSyQkhrI8x1vV2zm11pLGWO3p15plw0EED3uZehJJUWgxZ0jAMKvj853/Cn/6U3Zbn6JV7EUpS6TFkSUM0qOCTO3r17nfDtB4l4/rUV8FRSVLxcoNoaYjy2oT5ySd739R5EAEL3ItQkkqRI1nSEA1YaT03XB15JBx33JA+y4KjklR6fLtQGoZe12Q9/zz8v/+XfaFvDkrSuOTbhdIo6QpWXVOF9a+8kn3BW94CTu1JUlkyZEnD0LX4ff+2NtZXVrJ06VJqamrSJx29kqSy5sJ3aRiampo4e/duTkmlaGtv58EHH0yfMGBJUtkzZElDtWsXJ23dSipjXeMnfvlLmqdMKWCnJEnFwpAlDcXq1fDLX/L666/TtY3zXUBHR0fvpRwkSWXHNVnSYHR0wD33dB/W1tbyQHU1u9vbAaiqqrKGlSQJMGRJ+cute3XAAdQsWcKDZ5zBypUrAVi8eLE1rCRJgCFLyk9uwFqwgOb162latoyGhga++93vFqZfkqSiZciS+pMbrgAaG92wWZI0IBe+S33JDVhz53aXZshr30JJUllzJEvq1LVFzgVVVRx7zDHZJ3PqXg24b6EkqewZsiT2Vm6fn1u5fc4cOOSQHte7YbMkaSCGLAl44v/8H+a3tdGRSkEySWtrKzVLlvR7T319veFKktQn12RJq1dz2syZVFZWkqio4PfV1Uy/+upC90qSVOJCzNgSpBjU1dXFlpaWQndDJaRrLdWgp+2eew5aW7sPN23axH8kk07/SZLyFkJYH2Os6+2c04UqaUMupZD75uDs2dQ0NvKZ0emmJKkMOV2okjboUgo7d/YMWI2NcOiho9ZHSVJ5ciRLJW1QpRRyw9W0afDud49q/yRJ5cuQpZKWVymFtjb4xS+y23LqXkmSNNIMWSp5/ZZS6GNbHEmSRpshS+NTKgV3353ddu65EEJh+iNJKjuGLI0/jl5JkoqAIUvjS27AWrQIKrP/mA+5rpYkSYNgyFLJ6Dcc5Tl6NeS6WpIkDZIhSyWh33CUG7AWLIDq6l6f01tdLUOWJGk0GLJUEjLDUVtbG5///Of52oUXcsK0adkXDrD2alB1tSRJGgZDlkpCVzhqa2sjlUqxzwMP8PmmJpYuXUpNTQ38zd/AgQcO+Jy86mpJkjQCDFkqCV3h6Lbrr+dPjz1GKkaSySStra3ULFky6GcZriRJo829C1Uy6l95hf9+4YVUVVWRqKhgQ1UV06++utDdkiSpVyHGWOg+ZKmrq4stLS2F7oaKyZ/+BP/5n92HmzZt4j+SSaf7JEkFF0JYH2Os6+2c04UqbrlvDh54IDVLlvCZwvRGkqS8GbJUnHbvhgceyG6zarskqYQYslR8hrEtjtXcJUnFwpCl4tHRAffck93W2JgOTsuWDRicrOYuSSomhiwVTNao0yuv9LygM2DlG5ys5i5JKiaGLBVEd3hqa+OCysq9RUUBzj0XQgAGF5ys5i5JKiaGLBVEU1MT89va6Eil9hYVranpsfZqMMHJau6SpGJiyFJBXFhZyfrKSkgmqays5LArr4Qzz+xx3WCDk9XcJUnFwmKkGltr18IbbwDpoqKtra1Mv/pqg5EkqSRZjFRFYdNXv0prayu1tbXU1NRQ89GPUnPAAYXuliRJo8KQpdH30kts+v73ueGGG0h2Tg9e39REvQFLkjSOGbI0ujoLi7a2tpJMJvlNKsWfOzqYbXkFSdI4Z8jS6Ni+HR59tPuwtraWX0yYYHkFSVLZMGRp5OVui3PccdQ0NrLmjDN6fUvQrXAkSeNRXiErhLAQ+FcgAdwSY/xKzvkPA18FtnU2fTvGeEvnuQ6gtbN9S4zxvBHot4rRG2+k3x7MlFH3qrfyCm6FI0karwYMWSGEBHATcDawFVgXQlgVY9yYc+mPY4zX9vKIXTHGdw27pypuuaNX06fDSScNeJtb4UiSxqt8RrLmAE/HGDcDhBB+BJwP5IYslaM334Q1a7Lbcqq298etcCRJ41U+IWsa8HzG8VbglF6uuyiE8DfAU8CnYoxd9+wTQmgBksBXYow/z70xhHAVcBXAjBkz8u+9xly/mzpPmQKDHIVyKxxJ0ng1UgvfVwM/jDG2hRA+CqwA3tN5bmaMcVsI4QjgwRBCa4zxmcybY4w3AzdDuuL7CPVJI6xr/VRsa2N97qbOA4xe9be43a1wJEnjUT4haxtwWMbxdPYucAcgxrg94/AW4J8zzm3r/N/NIYQm4CQgK2SpNDQ1NfHOtjbeOsCmzrm6wllbWxuJRIJvf/vbXHXVVWPUa0mSCqMij2vWAUeHEGaFEKqBi4FVmReEEN6ecXge8ERn+6QQwoTO308BTse1XKUpleLCykqmV1aSqKigsrKS6R/9aF7rr5qammhrayOVSrFnzx6uueYampubx6DTkiQVzoAjWTHGZAjhWuB+0iUcbo0xbgghfBFoiTGuAq4LIZxHet3VDuDDnbcfCywPIaRIB7qv9PJWoord+vXwwgvU1NSwdOnSQW/q3NDQQCKRIJVKAZBKpXyLUJI07oUYi2sJVF1dXWxpaSl0N9QltzTDggVQXT3ox9x8881cc801pFIpJkyYYD0sSdK4EEJYH2Os6+2cFd/Vuw0bYPPmvcf77ANnnz3kx1111VXU1tb6FqEkqWwYstRT7uhVQwO85S3DfqxvEUqSyokhS3vljl7BoAqLSpKkvQxZSssdvTr9dDj44ML0RZKkccCQNc71VwQUgOeeg9bW7DZHryRJGjZD1jjWVQS0a1/AHm/03X03dJZVAKCuDt7+9p4PkiRJg5ZPMVKVqKamJtrb2+no6KC9vZ2mpqb0iZdfTk8PZgasxkYDliRJI8iRrHEsswhoIpGgoaEB7rsP9uzZe9Hs2XDooQXroyRJ45UjWeNIc3Mzy5Yty9qyJoQAwERg0q9/nR2wGhsNWJIkjRJHskpY5qJ2oMf6q6amJpLJJOfESCJzQ+c+1l4NuEhekiTlzZBVonIXtV966aU91l81nHkmFyQSJGOksrKS2traPt8cHHCRvCRJGhRDVonKXdQOUF1d3R2SLqyspGbnzu4NnY9atIiaSy7J+3lu4CxJ0vAYskpUQ0NDVqhavHgxixcvpmntWi6sqkpPCwI1NTXU/H//H3Suzcr3eV1TkJIkaWhCjLHQfchSV1cXW1paCt2NktBjDdUDD8Du3Xsv2H9/mDt36M+TJEn9CiGsjzHW9XbOkawSlrXhcm5h0XPOgYr8Xx41YEmSNLIMWaVuBDZ1dtG7JEkjz5BVynI3dX7f+yCRGPRjXPQuSdLIM2SVojffhDVrstuGsamzi94lSRp5hqxSc++9kEzuPV64EKqqhvXI+vr67uKlrsmSJGlkGLJKRUcH3HNP9+ETTz7Jz/fsoWHKlBEJRVmL6CVJ0rAZsopIn2/45ay9euwtb+E9n/ucC9UlSSpihqwi0esbfiefnJ4e7HLIITBnDmuXLXOhuiRJRS7/QkoaVblv+P3x3/4tO2CdcQbMmQPsXaieSCRIJBJs2bKF5ubmAvVckiT1xpBVJLqCU2VFBRckErzr2GPTJ447Lv3m4KRJ3dd2LVS/8sorCSHwb//2b8ybN4/m5maam5tZtmyZoUuSpAJzurBI1NfX0/zd7/LUmjXU1tam9x488USYMaPP65uamkgmk92jXytXrmTFihWu1ZIkqQgYsopBjHDXXZx48MGc+IEPwOGHQ23tgLfl1rcCXKslSVKRMGQV2iuvQObU3tFHQ01NXrfm1rcCskayLCoqSVLhGLIKKbM0w0EHpRe3hzCoR+TWt7KoqCRJxcGQVQi7d8MDD+w9Puoo6FroPkwWFZUkqTgYssZa5ujV/vvDWWdBhS95SpI03hiyxkp7O/z+93uP3/1umDatcP2RJEmjypA1FjJHr446Kr24vdL/6yVJGs/8m74Pfe4jOFjbt+/9/emnw8EHj+zzJUlSUTJk9aLXfQQHG4QefRT22Sc9LThrFo/t2MHa5cu7yyoM+/mSJKmoGbJ6kbuP4KCLev7mN3tHsGpraX79deYtXNgdqi699FKLhkqSNM75WlsvMjdgHlRRz+efh9deg+OP73oQVFX1CG3A0J4vSZJKRogxFroPWerq6mJLS0uhuzH4NVOZi9sbG3s8K3d6EHBNliRJJS6EsD7GWNfrOUPWML3xBuy3HzzyCOzcma57dcABPS5zobskSeNPfyHLNVlD1bmpMwAnnJB+c7CfLXGsxC5JUnkxZA1FKgXJ5N7jqVMHveegJEka3wxZg5E5erVwIcybBxMnFrZPkiSpKPl24WBs3rz396mUAUuSJPWpLEeyhvTm4JFHwrHHwp49cMwxTg9KkqR+lV3IGnQ1967SDM88kw5ZNTVj01FJklTSym66sLdq7r164gl4803oCmDnnOPolSRJylvZjWR1VXPvGsnqUW09lYK7707/fvt2OOOMHsVFJUmSBlJ2Iau+vp41a9b0vSarogKqqtJrr049tTCdlCRJJc+K7wDt7XD//en1VkcfPbafLUmSSlZ/Fd/Lbk1Wrzo3beb55wvbD0mSNG6U3XRhtz170vsNnn467L8/zJ8PEyYUuleSJGmcyGskK4SwMITwZAjh6RDCp3s5/+EQwp9DCP+v89cVGecuDSH8ofPXpSPZ+WF5/HF4/fW9BUYNWJIkaQQNOJIVQkgANwFnA1uBdSGEVTHGjTmX/jjGeG3OvQcDNwJ1QATWd967c0R6PxwnnQSHHAKHHVbonkiSpHEon5GsOcDTMcbNMcZ24EfA+Xk+fwHwQIxxR2ewegBYOLSujrBEwoAlSZJGTT4haxqQuSJ8a2dbrotCCP8VQrgjhNCVXvK6N4RwVQihJYTQ8uc//znPrkuSJBWvkXq7cDVweIzxnaRHq1YM5uYY480xxroYY91b3/rWEeqSJElS4eQTsrYBmfNq0zvbusUYt8cY2zoPbwFm53uvJEnSeJRPyFoHHB1CmBVCqAYuBlZlXhBCeHvG4XnAE52/vx+YH0KYFEKYBMzvbJMkSRrXBny7MMaYDCFcSzocJYBbY4wbQghfBFpijKuA60II5wFJYAfw4c57d4QQvkQ6qAF8Mca4YxR+DkmSpKLitjqSJElD5LY6kiRJY8yQJUmSNAoMWZIkSaPAkCVJkjQKDFmSJEmjwJAlSZI0CgxZkiRJo8CQJUmSNAoMWZIkSaPAkCVJkjQKim5bnRDCn4E/jtLjpwCvjNKzNXx+P8XN76e4+f0UN7+f4jac72dmjPGtvZ0oupA1mkIILX3tL6TC8/spbn4/xc3vp7j5/RS30fp+nC6UJEkaBYYsSZKkUVBuIevmQndA/fL7KW5+P8XN76e4+f0Ut1H5fspqTZYkSdJYKbeRLEmSpDFhyJIkSRoF4zJkhRAWhhCeDCE8HUL4dC/nJ4QQftx5/jchhMML0M2ylcf3899DCBtDCP8VQlgTQphZiH6Wq4G+n4zrLgohxBCCr6WPoXy+nxDCBzv/GdoQQvjBWPexnOXx77cZIYS1IYTfdf477n2F6Gc5CiHcGkJ4OYTw+z7OhxDCNzu/u/8KIbx7uJ857kJWCCEB3AQsAo4DLgkhHJdz2eXAzhjjUcDXgX8a216Wrzy/n98BdTHGdwJ3AP88tr0sX3l+P4QQ3gJ8AvjN2PawvOXz/YQQjgY+A5weYzwe+ORY97Nc5fnPzw3AT2KMJwEXA98Z216WtduBhf2cXwQc3fnrKuC7w/3AcReygDnA0zHGzTHGduBHwPk515wPrOj8/R3AvBBCGMM+lrMBv58Y49oY45udh48B08e4j+Usn39+AL5E+j9Odo9l55TX93MlcFOMcSdAjPHlMe5jOcvn+4nAAZ2/PxB4YQz7V9ZijA8DO/q55HxgZUx7DDgohPD24XzmeAxZ04DnM463drb1ek2MMQm8Ckwek94pn+8n0+XAvaPaI2Ua8PvpHEI/LMZ491h2TEB+//y8A3hHCOGREMJjIYT+/stdIyuf7+fzwIdCCFuBe4CPj03XlIfB/v00oMphdUcaRSGEDwF1wFmF7ovSQggVwL8AHy5wV9S3StLTHQ2kR4EfDiHUxhj/UshOqdslwO0xxq+FEOqB74cQTogxpgrdMY288TiStQ04LON4emdbr9eEECpJD9luH5PeKZ/vhxDCe4F/BM6LMbaNUd808PfzFuAEoCmE8BxwKrDKxe9jJp9/frYCq2KMe2KMzwJPkQ5dGn35fD+XAz8BiDE2A/uQ3pxYhZfX30+DMR5D1jrg6BDCrBBCNemFhatyrlkFXNr5+78FHoxWZR0rA34/IYSTgOWkA5brScZWv99PjPHVGOOUGOPhMcbDSa+ZOy/G2FKY7padfP799nPSo1iEEKaQnj7cPIZ9LGf5fD9bgHkAIYRjSYesP49pL9WXVcDizrcMTwVejTG+OJwHjrvpwhhjMoRwLXA/kABujTFuCCF8EWiJMa4C/p30EO3TpBfBXVy4HpeXPL+frwL7Az/tfB9hS4zxvIJ1uozk+f2oQPL8fu4H5ocQNgIdwJIYoyP1YyDP7+d64N9CCJ8ivQj+w/5H/tgIIfyQ9H+ATOlcE3cjUAUQY/we6TVy7wOeBt4EPjLsz/S7lSRJGnnjcbpQkiSp4AxZkiRJo8CQJUmSNAoMWZIkSaPAkCVJkjQKDFmSJEmjwJAlSZI0Cv5/bOoIOzYvbxoAAAAASUVORK5CYII="/>

<pre>
w: 0.3, b: 0.5
</pre>
샘플 데이터셋인 `x`와 `y`를 `torch.as_tensor()`로 텐서(Tensor)로 변환합니다.



```python
# 샘플 데이터셋을 텐서(tensor)로 변환
x = torch.as_tensor(x)
y = torch.as_tensor(y)
```

랜덤한 `w`, `b`를 생성합니다. `torch.rand(1)`은 `torch.Size([1])`을 가지는 normal 분포의 랜덤 텐서를 생성합니다.



```python
# random 한 값으로 w, b를 초기화 합니다.
w = torch.rand(1)
b = torch.rand(1)

print(w.shape, b.shape)

# requires_grad = True로 설정된 텐서에 대해서만 미분을 계산합니다.
w.requires_grad = True
b.requires_grad = True
```

<pre>
torch.Size([1]) torch.Size([1])
</pre>
다음은 가설함수(Hypothesis Function), 여기서는 Affine Function을 정의합니다.



```python
# Hypothesis Function 정의
y_hat = w * x + b
```

`y_hat`과 `y`의 손실(Loss)을 계산합니다. 여기서 손실함수는 **Mean Squared Error** 함수를 사용합니다.


<p>$\Large Loss = \sum_{i=1}^{N}(\hat{y}_i-y_i)^2$</p>



```python
# 손실함수 정의
loss = ((y_hat - y)**2).mean()
```

`loss.backward()` 호출시 미분 가능한 텐서(Tensor)에 대하여 미분을 계산합니다.



```python
# 미분 계산 (Back Propagation)
loss.backward()
```

`w`와 `b`의 미분 값을 확인합니다.



```python
# 계산된 미분 값 확인
w.grad, b.grad
```

<pre>
(tensor([-0.6570]), tensor([-1.1999]))
</pre>
## 경사하강법 구현



- 최대 500번의 iteration(epoch) 동안 반복하여 w, b의 미분을 업데이트 하면서, 최소의 손실(loss)에 도달하는 `w`, `b`를 산출합니다.

- `learning_rate`는 임의의 값으로 초기화 하였으며, `0.1`로 설정하였습니다.


하이퍼파라미터(hyper-parameter) 정의



```python
# 최대 반복 횟수 정의
num_epoch = 500

# 학습율 (learning_rate)
learning_rate = 0.1
```


```python
# loss, w, b 기록하기 위한 list 정의
losses = []
ws = []
bs = []

# random 한 값으로 w, b를 초기화 합니다.
w = torch.rand(1)
b = torch.rand(1)

# 미분 값을 구하기 위하여 requires_grad는 True로 설정
w.requires_grad = True
b.requires_grad = True

for epoch in range(num_epoch):
    # Affine Function
    y_hat = x * w + b

    # 손실(loss) 계산
    loss = ((y_hat - y)**2).mean()
    
    # 손실이 0.00005보다 작으면 break 합니다.
    if loss < 0.00005:
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
    ws.append(w.item())
    bs.append(b.item())

    if epoch % 5 == 0:
        print("{0:03d} w = {1:.5f}, b = {2:.5f} loss = {3:.5f}".format(epoch, w.item(), b.item(), loss.item()))
    
print("----" * 15)
print("{0:03d} w = {1:.1f}, b = {2:.1f} loss = {3:.5f}".format(epoch, w.item(), b.item(), loss.item()))
```

<pre>
000 w = -0.01099, b = 0.89355 loss = 0.10401
005 w = -0.05946, b = 0.74602 loss = 0.01675
010 w = -0.05314, b = 0.70280 loss = 0.01087
015 w = -0.03533, b = 0.68401 loss = 0.00940
020 w = -0.01588, b = 0.67136 loss = 0.00829
025 w = 0.00302, b = 0.66061 loss = 0.00733
030 w = 0.02093, b = 0.65077 loss = 0.00648
035 w = 0.03779, b = 0.64158 loss = 0.00573
040 w = 0.05365, b = 0.63295 loss = 0.00506
045 w = 0.06856, b = 0.62485 loss = 0.00448
050 w = 0.08257, b = 0.61724 loss = 0.00396
055 w = 0.09573, b = 0.61008 loss = 0.00350
060 w = 0.10811, b = 0.60336 loss = 0.00310
065 w = 0.11974, b = 0.59704 loss = 0.00274
070 w = 0.13067, b = 0.59110 loss = 0.00242
075 w = 0.14094, b = 0.58551 loss = 0.00215
080 w = 0.15060, b = 0.58027 loss = 0.00190
085 w = 0.15967, b = 0.57534 loss = 0.00168
090 w = 0.16820, b = 0.57070 loss = 0.00149
095 w = 0.17622, b = 0.56634 loss = 0.00132
100 w = 0.18375, b = 0.56225 loss = 0.00117
105 w = 0.19083, b = 0.55840 loss = 0.00104
110 w = 0.19749, b = 0.55479 loss = 0.00092
115 w = 0.20374, b = 0.55139 loss = 0.00082
120 w = 0.20962, b = 0.54819 loss = 0.00073
125 w = 0.21514, b = 0.54519 loss = 0.00065
130 w = 0.22034, b = 0.54237 loss = 0.00057
135 w = 0.22522, b = 0.53972 loss = 0.00051
140 w = 0.22980, b = 0.53722 loss = 0.00046
145 w = 0.23411, b = 0.53488 loss = 0.00041
150 w = 0.23817, b = 0.53268 loss = 0.00036
155 w = 0.24197, b = 0.53061 loss = 0.00033
160 w = 0.24555, b = 0.52866 loss = 0.00029
165 w = 0.24892, b = 0.52684 loss = 0.00026
170 w = 0.25208, b = 0.52512 loss = 0.00024
175 w = 0.25505, b = 0.52350 loss = 0.00021
180 w = 0.25784, b = 0.52199 loss = 0.00019
185 w = 0.26047, b = 0.52056 loss = 0.00017
190 w = 0.26293, b = 0.51922 loss = 0.00016
195 w = 0.26525, b = 0.51796 loss = 0.00014
200 w = 0.26743, b = 0.51678 loss = 0.00013
205 w = 0.26948, b = 0.51566 loss = 0.00012
210 w = 0.27140, b = 0.51462 loss = 0.00011
215 w = 0.27321, b = 0.51363 loss = 0.00010
220 w = 0.27491, b = 0.51271 loss = 0.00009
225 w = 0.27651, b = 0.51184 loss = 0.00009
230 w = 0.27801, b = 0.51103 loss = 0.00008
235 w = 0.27942, b = 0.51026 loss = 0.00008
240 w = 0.28075, b = 0.50954 loss = 0.00007
245 w = 0.28200, b = 0.50886 loss = 0.00007
250 w = 0.28317, b = 0.50822 loss = 0.00006
255 w = 0.28427, b = 0.50762 loss = 0.00006
260 w = 0.28530, b = 0.50706 loss = 0.00006
265 w = 0.28628, b = 0.50653 loss = 0.00005
270 w = 0.28719, b = 0.50604 loss = 0.00005
275 w = 0.28805, b = 0.50557 loss = 0.00005
------------------------------------------------------------
277 w = 0.3, b = 0.5 loss = 0.00005
</pre>
## 결과 시각화



- `loss`는 epoch이 늘어남에 따라 감소합니다.

-  epoch 초기에는 급격히 감소하다가, 점차 완만하게 감소함을 확인할 수 있는데, 이는 초기에는 큰 미분 값이 업데이트 되지만, 점차 계산된 미분 값이 작아지게되고 결국 업데이트가 작게 일어나면서 손실은 완만하게 감소하였습니다.

- `w`, `b`도 초기값은 `0.3`, `0.5`와 다소 먼 값이 설정되었지만, 점차 정답을 찾아가게 됩니다.



```python
# 전체 loss 에 대한 변화량 시각화
plt.figure(figsize=(14, 6))
plt.plot(losses, c='darkviolet', linestyle=':')

plt.title('Losses over epoches', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# w, b에 대한 변화량 시각화
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(14, 6)

axes[0].plot(ws, c='tomato', linestyle=':', label='chages')
axes[0].hlines(y=0.3, xmin=0, xmax=len(ws), color='r', label='true')
axes[0].set_ylim(0, 0.7)
axes[0].set_title('"w" changes over epoches', fontsize=15)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Error')
axes[0].legend()

axes[1].plot(bs, c='dodgerblue', linestyle=':', label='chages')
axes[1].hlines(y=0.5, xmin=0, xmax=len(ws), color='dodgerblue', label='true')
axes[1].set_ylim(0.2, 0.9)
axes[1].set_title('"b" changes over epoches', fontsize=15)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Error')
axes[1].legend()

plt.show()
```

<img src="data:image/png;base64,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"/>

<img src="data:image/png;base64,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br81M7cf+Gduz5zRoRgefvjhNGbMmDR9+vSUUkozZsxIRx99dPrIRz6SlixZkiZPnpw23HDDIqRF6c0330wppTR9+vS04YYbpqampvSvf/0rjR07Ns2bNy+99dZbaaONNkpnnXVWSimlvfbaKz3xxBMppZTuvvvu9L73vS+llNKWW26ZpkyZklJK6Y033ujY55NSAiamGsgJtfhY1Tz16uyU5i1apU1IqkOtj8WHX5HS5ZPz84WLc/t/H83tuQtze8Ljuf3m/Ny+9sncnjE3t//+dG5Pm93++9dTnnKUsSTVgJtuuonDDjuMtdZaC4A111wTgEMOOYQePXqw+eabM21anhospcS3vvUtbrvtNnr06MFLL73EtGnTuPPOOzn44IPp168f/fr148ADDwTymbi77rqLww477O33W7BgAQC77LILxxxzDIcffjgf/vCHu3KXtQzDBpQdgSS9Uz3lKQsgSWrtxLNanvfqtXS7b7+l2/0HLN0etPrS7dXXXKVQ+vZtuStmPqEFF198MdOnT+e+++6jd+/ejBo1arlTgDY1NTFkyBAmTZr0jtd+9atfcc8993DNNdew/fbbc9999zF06NBVilmr5q9P5H8P3LjcOCTVtss+0vK8d8+l26v1Xro9uO/S7TVXW7q99iqceOmOecprgCSpBuy1115cccUVzJgxA4DXX399meu++eabrL322vTu3Zubb76Z559/Hshnyf76178yf/58Zs+ezdVXXw3A4MGDGT16NFdccQWQE9SDDz4I5DHXO+20E6effjrDhg3jxRdfrOZuqgP++BD84aGyo5CkpdVTnrIHSJJqwBZbbMEpp5zCHnvsQc+ePdl2222Xue7HPvYxDjzwQLbaaivGjRvHpptuCsAOO+zAQQcdxNZbb83w4cPZaqutWH311YF8Nu6LX/wi3//+91m0aBFHHnkkY8eO5YQTTuDJJ58kpcTee+/N2LFju2R/tWy/Owh6e3pSUo2ppzwVzV1V3cW4cePSxIkTyw5DUh159NFH2WyzzcoOo1PMnj2bgQMHMnfuXHbffXfOO+88tttuu1XaZlufT0Tcl1Iat0obrlPmKUnVUC+5qhbylD1AklRHjj32WB555BHmz5/P0UcfvcpJRV3vjhfg4VfhC5aXkupQLeQpCyBJqiOXXHJJ2SFoFd3+AlwwyQJIUn2qhTzlKGNJ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"/>
