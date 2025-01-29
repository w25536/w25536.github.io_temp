---
layout: page
title: "논문 리뷰 : Adding Conditional Control to Text-to-Image Diffusion Models"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---
---

## **Contents**

[1. Simple Introduction](#1)

[2. Background Knowledge: Diffusion]()

[3. Method]()

[4. Result]()


---

## LDM (Stable Diffusion) Structure[#1]


![](https://cdn.mathpix.com/snip/images/g6wV3-r-LChmmQjW3RwFd5Xu3Ti1qGblKFbk-ZYaKzM.original.fullsize.png)



- 기존 Stable Diffusion LDM에서 컨디셔닝을 어떻게 처리하는지 살펴보겠음.
- 일단, 다양한 방법중에 널리 알려진 기능 텍스트(프롬프트)를 통한 Text-to-Image 생성
- Image-to-Image 변환 방식에 존재 
- ControlNet 방법이 쉬우면서도  근본적인 문제를 극복 할 수 있는 모델이라고 필자는 설명
- 



![](https://cdn.mathpix.com/snip/images/VoYPN658c6bfbc-MWSrbW9Y-vBzRtfQOaa-LmB7KdLc.original.fullsize.png)

-  한마디로 위그림처럼 x의 feature를 받아서 y라는 출력값을 생성할 때, ControlNet 모듈의 컨디션 c를 추가로 고려하여 최종적으로 y 값을 내도록 유도


![](https://cdn.mathpix.com/snip/images/2iNLaVfzFVsBGPqYS-NgQp6GYM7n2VpgYNNjT4v2qAU.original.fullsize.png)

> [!NOTE]
> Control pretrained large diffusion models to support additional input conditions
> 


- 위 이미지를 조금더 구체적으로 이해해보면 "pretrained larged diffusion models" 가져와 활용함
-  Training을 시킬때 frozen 을 시켜서 기존의 large 모델에 장점을 최대한 살리고
- 추가적인 c 컨디션을 붙여 다른 모델도 함께 학습하는 방식으로 기존 대형 모델의 장점을 최대한 살린다고 설명 한다
- ControlNet은 정말 간단한 적용만으로도, 쉽게 diffusion network를 통제할 수 있도록 설계되어 있음. 

바로 위의 그림처럼, network를 기준으로, 아래 2개의 개념을 적용

1. zero-convolution
2. trainable copy

각각이 무엇인지 한번 살펴보자.



첫번째 부터  zero convolution layer 통과할때 
- weight bias모두 zero 로 intiailize 된 1x1 conv layer
- 학습이 진행되면서 최적화된 파라미터를 가질 수 있고  
- 왜 weight와 bias 를 모두 0를 초기화 하는 이유는 
- 아래에 이미지 확인 할 수 있다. 


![](https://cdn.mathpix.com/snip/images/48hMZkKzpNTRI-p5QMDH1s6gPNbupxDRMJr0a940aqE.original.fullsize.png)


![](https://cdn.mathpix.com/snip/images/AELbV6cfab3P5IsRuAgEwCEcWc5o17VRhw3qpQHKogM.original.fullsize.png)



먼저 trainable copy가 무엇인지 정의를 하고 가자.

기존의 input feature map X와 layer에 해당하는 파라미터 Θ가 있다고 하고, 이를 함수 F로 표현하자.

그렇다면, trainable copy는 함수 F에 해당하는 구조와 파라미터 Θ를 그대로 복제한 layer 이다.




pretrained large diffusion models 


y 


c -> conditon 


yc -> contolnet 



추가적으로 학습해서 붙친다고 이해하시면 될꺼 같습니다 



zero convolution 

처음에 weight bias모두 zero 로 intiailize 된 1x1 conv layer

trainable copy  전달 할때 noise가 있을때 condition이 망가 질수 있기 때문에 그렇다 



![](https://cdn.mathpix.com/snip/images/17_LDq4pTZ2fpfQ9pDBy_g9RkQBLTXSP_ovGyEN1Fi4.original.fullsize.png)






초기 학습에서 random noise를 제거하여서 학습이 올바른 방향으로 진행될 수 있도록 유도하였다고 설명하고 있다.



![](https://cdn.mathpix.com/snip/images/17_LDq4pTZ2fpfQ9pDBy_g9RkQBLTXSP_ovGyEN1Fi4.original.fullsize.png)

$$ 
\begin{aligned}&y=w x+b\\&\partial y / \partial w=x, \partial y / \partial x=w, \partial y / \partial b=1\\&\text { if } w=0 \text { and } x \neq 0\\& \partial y / \partial w \neq 0, \partial y / \partial x=0, \partial y / \partial b \neq 0\end{aligned}
$$




![](https://cdn.mathpix.com/snip/images/QTON_AlO7DZSwYEf-jhaOc1sX-6WmXrA4qCdjB2TwAs.original.fullsize.png )


## First Training Step 



$$
y_c=\mathcal{F}(x ; \Theta)+\mathcal{Z}\left(\mathcal{F}\left(x+\mathcal{Z}\left(c ; \Theta_{z 1}\right) ; \Theta_c\right) ; \Theta_{z 2}\right)
$$


Zero convolution layer의 가중치와 바이어스는 모두 0으로 초기화되기 때문에, 첫 번째 학습 step에서 다음과 같다.


i) 각각의 component로 편미분 했을때 weight가 0여도 편미분한 값이 0이 아니기 때문에 $W^*$ 값이 0이 아닌수로 값이 바뀌게 된다. 


### i)  Foward pass:


$$
\mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}=\boldsymbol{B}_i+\sum_j^c \boldsymbol{I}_{p, i} \boldsymbol{W}_{i, j} \quad\left\{\begin{array}{l}
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{B}_i}=1 \\
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{I}_{p, i}}=\sum_j^c \boldsymbol{W}_{i, j}=0 \\
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{W}_{i, j}}=\boldsymbol{I}_{p, i} \neq \mathbf{0}
\end{array}\right.

$$




### ii)
$$
W^*=W-\beta_{\mathrm{lr}} \cdot \frac{\partial \mathcal{L}}{\partial \mathcal{Z}(I ;\{W, B\})} \odot \frac{\partial \mathcal{Z}(I ;\{W, B\})}{\partial W} \neq 0
$$



<style type="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // 'code' 항목 제거
    },
    "HTML-CSS": {
        availableFonts: ["TeX"]
    },
    TeX: {
        extensions: ["AMSmath.js", "AMSsymbols.js"]
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

