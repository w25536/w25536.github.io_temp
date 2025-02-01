---
layout: page
title: Wall Segmentation 및 실험
description: ""
headline: ""
tags:
  - python
  - 파이썬
  - pytorch
  - 파이토치
  - 전처리
  - science
  - 딥러닝
categories: deep-learning
comments: true
published: true
---

# Wall Segmentation 실험


아래의 논문을 참고함 :

- [Getting Started with Wall Segmentation](https://www.researchgate.net/publication/363059238_Getting_Started_with_Wall_Segmentation)
- [Wall segmentation in 2D images using convolutional neural networks](https://www.researchgate.net/publication/373861585_Wall_segmentation_in_2D_images_using_convolutional_neural_networks)
- https://arxiv.org/pdf/1612.01105

Segmentation architecture :

Segmentation 모델의 구조는 크게 Encoder와 Decoder로 구성



## 실험 환경


## 데이터셋

실험에 사용된 데이터셋은 150개의 서로 다른 카테고리가 라벨링된 [MIT ADE20K Scene parsing dataset](http://sceneparsing.csail.mit.edu/)입니다. 데이터베이스의 이미지 예시:


![](https://cdn.mathpix.com/snip/images/6SmSJp9CmmPLoM1urb_s3BK15xboUDHUQwR4AXYIlOs.original.fullsize.png)


### 인코더
- Dilated ResNet-50을 기반으로 구성
- 원본 이미지의 특징을 추출하여 더 작은 높이와 너비를 가지고 채널 수가 증가된 feature map을 생성
- ResNet-50의 마지막 두 블록에서는 dilated convolution을 사용하여 stride를 줄임
- 이를 통해 receptive field를 키우면서도 spatial resolution 손실을 최소화

### 디코더
- PPM(Pyramid Pooling Module) 구조 채택
- Encoder에서 추출된 feature map을 바탕으로 각 픽셀을 해당하는 클래스로 분류
- 다양한 스케일의 context 정보를 효과적으로 활용하여 segmentation 성능 향상


## 학습 파라미터

이 프로젝트의 주요 기여는 기존 [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) 프로젝트를 벽 분할에 특화된 형태로 단순화하는 것입니다. 모델 학습을 위해 다음 세 가지 접근 방식을 시도했습니다:

1. 전이 학습 + 출력 레이어 재학습
   - ImageNet 사전학습된 ResNet-50 인코더 사용
   - 150개 카테고리에 대해 전체 모델 학습 
   - 디코더의 출력 레이어만 벽 분할을 위해 재학습

2. 전이 학습 + 디코더 전체 재학습  
   - ImageNet 사전학습된 ResNet-50 인코더 사용
   - 150개 카테고리에 대해 전체 모델 학습
   - 디코더 전체를 벽 분할을 위해 재학습

3. End-to-End 학습
   - ImageNet 사전학습된 ResNet-50 인코더 사용
   - 전이 학습 없이 벽 분할을 위해 처음부터 학습

추가로 다른 백본 네트워크를 사용한 실험도 진행했습니다:
- Dilated ResNet-50 기반 End-to-End 학습


## 결과 요약

Transfer learing - entire decoder 



    6.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_1.png)
    


    7.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_3.png)
    


    8.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_5.png)
    


    5.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_7.png)
    


    9.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_9.png)
    


    4.jpg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_11.png)
    


    2.jpg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_13.png)
    


    1.jpg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_15.png)
    


    3.jpeg
    Building encoder: resnet50-dilated
    Loading weights for net_encoder
    Loading weights for net_decoder



    
![png](2025-02-01/testing_4_17.png)





## 다음 실험 계획

1. 세크멘테이션 고도화 
2. OpenCV를 이용하여 처리된 결과의 wall segmentation을 고도화 시키기

