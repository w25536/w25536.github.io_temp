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

이전 Vision Task에서 Self-Attention적용의 한계
      - Self-Attention을 적용하는 시도는 있었으나, Hardware Accelerators에 비효율적

          → ResNet구조가 SOTA였음

     ▶ 기존의 Transformer를 최대한 그대로 적용하고자 함

Attention is All you Need
      - NLP에서 가장 대표적인 구조 "Self-Attention"를 활용한 Transformer

      - 대표 모델 "BERT"는 Large Dataset(Corpus)를 사전학습(Pre-Train) → 작은 Task에서 미세조정(Fine-Tune) 

Transformer의 장점
      - Transfoerm의 계산 효율성(Efficiency) 및 확장성(Scalability)

      - 100B Parameter도 학습 가능!

      - 데이터셋이 크면 클수록 모델을 키워도 되며, 성능이 포화(Saturate)될 징후 X
      - 더 키우면 키울수록 더 성능은 높아질 것임!

Transformer의 적용 방안
      - 이미지를 Patch로 분할 후 Sequence로 입력

        → NLP에서 단어(Word)가  입력되는 방식과 동일! ( ∵ 논문 제목이 "IMAGE IS WORTH 16X16 WORDS"인 이유)

      - Supervised Learning 방식으로 학습

Transformer의 특징
      - ImageNet와 같은 Mid-sized 데이터셋으로 학습 시, ResNet보다 낮은 성능을 보임

        ( * ImageNet은.. 더 이상 큰 데이터셋이 아니다..)

      - JFT-300M 사전 학습 후, Transfer Learning → CNN구조 보다 매우 좋은 성능 달성(SOTA)

      - Transformer는 inductive biases가 없음 = Locality와 Translation Equivariance 같은 CNN의 특성이 없음
