---
layout: page
title: " Transformers :AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE 논문 분석"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---


<https://iy322.tistory.com/66>

Vision Transformer (ViT)의 계산 과정을 자세히 설명하겠습니다.

## 입력 이미지 처리 (Z_0)

1. 이미지 패치화:

    - 입력 이미지 (H x W x C)를 N개의 패치로 분할합니다[1](https://hyundoil.tistory.com/334).
    - 각 패치의 크기는 (P x P x C)입니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).
    - 예: 224x224x3 이미지를 16x16 패치로 나누면 14x14=196개의 패치가 생성됩니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

2. 패치 임베딩:

    - 각 패치를 1차원 벡터로 평탄화(flatten)합니다[1](https://hyundoil.tistory.com/334).
    - 평탄화된 벡터에 선형 변환을 적용하여 D 차원의 임베딩 벡터로 변환합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).
    - 예: 16x16x3=768 차원의 벡터를 D 차원으로 변환합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

3. 위치 임베딩 추가:

    - 학습 가능한 1D 위치 임베딩을 각 패치 임베딩에 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).
    - 위치 임베딩의 크기는 (N+1) x D입니다 (클래스 토큰 포함)[5](https://velog.io/@leehyuna/Vision-TransformerViT).

4. 클래스 토큰 추가:

    - 학습 가능한 클래스 토큰을 패치 임베딩 시퀀스의 맨 앞에 추가합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

## Transformer 인코더 처리

5. 레이어 정규화 (Layer Normalization):

    - 입력 시퀀스에 레이어 정규화를 적용합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

6. 멀티헤드 셀프 어텐션 (Multi-Head Self-Attention, MSA):

    - 정규화된 입력을 Query, Key, Value로 변환합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).
    - 각 헤드에서 어텐션 스코어를 계산: A = softmax(QK^T / √d)[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).
    - 어텐션 스코어와 Value를 곱하여 컨텍스트 벡터를 생성합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).
    - 모든 헤드의 결과를 연결하고 선형 변환을 적용합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog).

7. 잔차 연결 (Residual Connection):

    - MSA의 출력을 입력과 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

8. 레이어 정규화:

    - 잔차 연결의 결과에 다시 레이어 정규화를 적용합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

9. MLP (Multi-Layer Perceptron):

    - 정규화된 결과를 MLP에 통과시킵니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).
    - MLP는 일반적으로 두 개의 선형 레이어와 GELU 활성화 함수로 구성됩니다.

10. 잔차 연결:

    - MLP의 출력을 이전 단계의 입력과 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

11. 반복:

    - 5-10 단계를 L번 반복합니다 (L은 Transformer 블록의 수)[5](https://velog.io/@leehyuna/Vision-TransformerViT).

## 분류 헤드

12. 최종 레이어 정규화:

    - 마지막 Transformer 블록의 출력을 정규화합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

13. 클래스 토큰 추출:

    - 정규화된 출력에서 클래스 토큰에 해당하는 벡터를 추출합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

14. 분류:

    - 추출된 클래스 토큰 벡터를 선형 레이어에 통과시켜 최종 분류 결과를 얻습니다[5](https://velog.io/@leehyuna/Vision-TransformerViT).

이 과정을 통해 ViT는 이미지를 패치 단위로 처리하고, Transformer 구조를 사용하여 전역적인 관계를 학습하며, 최종적으로 이미지 분류를 수행합니다.

Downstream 이란 무엇인가 예측 하는ㄷ

positonal encoding

0실게 16 by 16 나눈다 flatten 시켜서

embeding vector가 되고

정수로 postional encoding을 진행

class token이 쓰여진 이유 ?

분류 task이니까 정답을 같이 준거고

0에서는 클래스 토큰이다.

그 클래스 정보를 어덯게 학습 하나?

64 차원을 학읍 하고 vitT에서 어떻게 학습 되는?

MLP clas

clasification token 만쓴다.

BERT 모델
