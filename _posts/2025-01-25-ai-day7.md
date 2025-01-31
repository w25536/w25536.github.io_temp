---
layout: page
title: "ai day7"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---
## Cross Attention 계산 과정

1. **입력 준비**

    - Query (Q): decoder의 이전 layer 출력
    - Key (K): encoder의 최종 출력
    - Value (V): encoder의 최종 출력

2. **Attention Score 계산**

    - Attention Score = (Q * K^T) / sqrt(d_k)
    - 여기서 d_k는 key 벡터의 차원

3. **Softmax 적용**

    - Attention Weight = softmax(Attention Score)

4. **최종 출력 계산**

    - Output = Attention Weight * V

## 구현 단계

1. **Query, Key, Value 생성**

    - Q = decoder_output * W_q
    - K = encoder_output * W_k
    - V = encoder_output * W_v
    - W_q, W_k, W_v는 학습 가능한 가중치 행렬

2. **Scaled Dot-Product Attention**python

 ```python
 def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(attention_score, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
  
 ```

3. **Multi-Head Attention**

    - 여러 개의 attention head를 병렬로 계산
    - 각 head의 결과를 concatenate한 후 linear projection

## Cross Attention의 특징

- Query는 decoder에서, Key와 Value는 encoder에서 오므로 서로 다른 입력 시퀀스 간의 관계를 학습할 수 있습니다[1](https://brunch.co.kr/@leadbreak/10)[5](https://kongsberg.tistory.com/47).
- encoder의 전체 입력 시퀀스 정보를 decoder의 각 단계에 반영할 수 있습니다[12](https://skyil.tistory.com/256).
- 번역, 요약 등의 작업에서 소스 언어와 타겟 언어 간의 관계를 효과적으로 포착할 수 있습니다[9](https://dsbook.tistory.com/399).

Cross attention은 encoder의 정보를 decoder가 효과적으로 활용할 수 있게 해주는 핵심 메커니즘으로, Transformer 모델의 성능 향상에 크게 기여합니다.


유도 편향 (Inductive Bias)

- CNN: 필터링을 통해 지역적 유도 편향 제공
- Self-Attention: 전역적 유도 편향 제공
- 그러므로, inductive bias 

