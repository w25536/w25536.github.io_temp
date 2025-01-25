---
layout: page
title: "AI 심화 day6"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---



Transformer 이해

Non-vision specific model
-Typically applied to 1-D sequence data

Transformer "encoder"

- A stack of alternating self-attention and MLP blocks
- Residual and LayerNorm

Transformer "decoder"

- A slightly more involved archetecture useful when the ouput space is different from the input space

![alt text](http://jalammar.github.io/images/t/transformer_decoding_2.gif)

## Q, K, V 어텐션 메커니즘

- **Q (Query)**: 디코더의 숨겨진 상태로, 현재 처리 중인 단어에 대한 정보를 담고 있습니다. (decoder 의 숨겨진 상태)
- **K (Key)**: 인코더의 숨겨진 상태로, 입력 시퀀스의 각 단어에 대한 정보를 담고 있습니다. (encoder 의 숨겨진 상태)
- **V (Value)**: 인코더의 숨겨진 상태로, 실제 어텐션 결과에 사용되는 값입니다. (encoder 의 숨겨진 상태)

어텐션 메커니즘은 Query(Q)와 Key(K) 간의 유사도를 계산하여 해당 Value(V)에 가중치를 부여합니다. 이를 통해 디코더는 입력 시퀀스의 중요한 부분에 집중할 수 있습니다.

특히, Q와 K의 내적(dot product)을 통해 어텐션 점수를 산출하고, 이를 소프트맥스(softmax) 함수를 이용해 확률로 변환합니다. 높은 어텐션 값은 높은 연관성을 나타내며, 해당 Value에 더 큰 영향을 미칩니다.

encoder 하는 역활은

context vector을 LSTM에 전달하고, 디코더를 학습시킬 때 teacher forcing을 적용합니다. 디코더는 단순히 context vector만 사용하는 대신, 인코더의 모든 은닉 상태를 참조하여 각 단계에서 중요한 정보를 선택적으로 활용할 수 있습니다. 이러한 방식이 바로 어텐션(attention) 기법입니다. 어텐션 메커니즘을 통해 디코더는 입력 시퀀스의 전체 정보를 고려하여 더 정확하고 효과적인 출력을 생성할 수 있습니다.

Self attention  왜? K 에서 부터 encoder 하닌깐

layer by norm 이  n 번 반복 된다.

Q, K, V 를 학습한다

multi-head가 왜 붙었어? head가 여려개 있다.

Convolutional Neural Networks (CNNs)은 컴퓨터 비전 분야에서 핵심적인 역할을 담당하는 신경망 구조로, 입력 이미지로부터 공간적 계층 구조의 특징을 자동적이고 적응적으로 학습하는 데 뛰어납니다. CNN은 필터(커널)를 사용하여 입력 데이터에 걸쳐 합성곱 연산을 수행함으로써 지역 패턴을 효과적으로 포착하며, 이를 통해 이미지 분류, 객체 검출, 세분화 등 다양한 작업에서 우수한 성능을 발휘합니다.

반면에 Transformer는 원래 자연어 처리(NLP) 분야에서 도입된 구조로, 주의 메커니즘(attention mechanism)을 활용하여 시퀀스 데이터 내의 장기적인 의존성과 문맥 관계를 모델링합니다. 자기 주의(Self-Attention) 메커니즘을 통해 Transformer는 입력 데이터의 서로 다른 부분 간의 중요도를 동적으로 평가할 수 있으며, 이는 전통적인 CNN이나 순환 신경망(RNN)에 비해 더욱 유연하고 강력한 표현력을 제공합니다.

교수의 시각에서 볼 때, CNN과 Transformer의 통합은 두 아키텍처의 강점을 동시에 활용하려는 시도라 할 수 있습니다. CNN은 지역적이고 공간적인 특징 추출에 탁월한 반면, Transformer는 글로벌한 관계와 문맥 정보를 효과적으로 캡처할 수 있습니다. 이러한 하이브리드 접근 방식은 이미지 인식 및 컴퓨터 비전 작업에서 CNN의 특징 추출 능력과 Transformer의 문맥 인식 능력을 결합함으로써 더욱 정교하고 정확한 모델을 구현할 수 있게 합니다. 예를 들어, CNN을 통해 추출된 특징 맵을 Transformer의 주의 메커니즘과 결합하여 복잡한 이미지 내의 다양한 객체 간의 관계를 이해하고, 이를 기반으로 보다 정밀한 예측을 가능하게 합니다.

attention ?
 input에 대해서 뭐가 중요하는지 weight를 정해는 얘다

- multi-head 정보를 모두 Feed Forward Neural Network (FFNN)을 통해 처리 (보통 FFNN이라고 칭함)
- FFNN의 역할: 각 어텐션 헤드에서 추출한 특징을 통합하여 다음 층으로 전달
- Multi-head는 여러 개의 어텐션 헤드를 의미
- 각 헤드는 서로 다른 행렬을 사용하여 입력 데이터의 다양한 측면을 학습
- 이러한 다중 헤드 구조는 다양한 표현 공간에서 정보를 효과적으로 처리

PE = E + P

### Sequence-to-Sequence (Seq2Seq) 모델

**Seq2Seq 모델이란?**  
Seq2Seq 모델은 입력 시퀀스를 다른 시퀀스로 변환하는 인공 신경망 구조입니다. 주로 기계 번역, 음성 인식 등 다양한 자연어 처리 작업에 사용됩니다.

#### Attention 기법

Attention 메커니즘은 Seq2Seq 모델의 성능을 향상시키기 위해 도입되었습니다. 전통적인 Seq2Seq 모델은 인코더의 마지막 숨겨진 상태만을 디코더에 전달하기 때문에, 긴 시퀀스의 경우 장기 의존성을 파악하는 데 한계가 있었습니다. 특히, vanishing gradient 문제로 인해 멀리 떨어진 단어 간의 관계를 학습하지 못하는 이슈가 발생했습니다.

**Attention 메커니즘의 역할:**  

- **장기 의존성 문제 해결:** Attention은 인코더의 모든 숨겨진 상태를 고려하여, 디코더가 각 단계에서 중요한 정보를 선택적으로 활용할 수 있게 합니다.
- **Positional Encoding:** 시퀀스의 순서를 보존하기 위해 입력에 positional encoding을 추가합니다. 이를 통해 모델은 단어의 순서 정보를 충분히 습득할 수 있습니다.

#### Q, K, V 개념

Attention 메커니즘에서 Query(Q), Key(K), Value(V)는 중요한 역할을 합니다.

- **Q, K, V Embedding:** 입력 시퀀스의 각 단어는 Q, K, V로 임베딩됩니다.
- **Attention 가중치 계산:** Query(Q)와 Key(K)의 내적을 통해 유사도를 계산하고, 이를 소프트맥스(softmax) 함수를 통해 확률로 변환합니다.
- **가중합:** 소프트맥스 결과에 Value(V)를 곱하여 최종적인 값(Value)을 얻습니다. 이 값을 통해 중요한 정보에 더 큰 가중치를 부여할 수 있습니다.

#### Softmax와 Value의 결합

Softmax를 통해 계산된 가중치는 Value에 곱해져 최종적인 값이 도출됩니다. 이는 디코더가 입력 시퀀스의 중요한 부분에 집중할 수 있도록 도와줍니다.

### 결론

Attention 메커니즘은 Seq2Seq 모델의 장기 의존성 문제를 효과적으로 해결하며, 더 정밀하고 효과적인 시퀀스 변환을 가능하게 합니다.

identity mapping

분산이 바뀐다

MHA, MHSA 둘다 비슷하다

encoder-decoder attention

vit_t e

encoding 만쓰는 얘들 bert, vision transformer

encoder + decoder = detection transformer

GPT 계열은 decoder 만이 있은 얘들이다 .

encoder 정보를 압축해서 그 정보로 부터

CNN 어떻게 정보를 어떻게 압축해 ? CNN은 이미지 사이즈가 줄면서 채널 정보를 늘리고 채널 정보가 분류 할때 차원이 된다.

transformer 같은 경우리 self attention n번 반복한다 점점 정재가 된다.

encode 쓸댸는 분류 할때만 쓴다.

BERT (Bidirectional Encoder Representations from Transformers)

Encoder 만 쌓아 양방향 문맥을 학습
다양한 다운스트림 태스크(분류, QA)에 파인 튜닝 가능

"BERT 사전 학습"

오레그레시브 학습 (이전)

few shot learning 하기 위해서 language model 사용

language model 뭐냐? language model 나는 나올 수 있는 다언의 위치 정보의 확율을 가지고 있다.

인코더 + 디코더를 모두 쓰는 전통적인 seq2seq 구조

T5, BART, Marian (구글 원 논문의 Transformer)등.

작동 원리

multi-head attention 다른 문맥들의 모델을 학습한다.

Position Encoding을 꼭 사인/코사인으로 해야 하나요?

- 필수는 아니다

RNN 기반 seq2seq -> Attention 도입 -> Q, K, V  -> Transformer (Encoder-Decoder) ->

Gradient checking

시퀀스 길어지면 Attention의 연산량이 O(n^2)이므로 비효울
Longformer, Bigbird, Performer 등에서 (Huggingface) 가져와서 써라.

Recurrent 에서ㄱ

Solution : Attention  is all you need !!!

Q, K, V  같은 얘들

weight 가 다른 얘들

True understanding the material vs memorization an pattern -matching

있는 정보들

연구진이 실제 연구를 어떻게 진행 했는지

experimental trakcing

MLP multiple layer perceptron

VIT (Encoder)
