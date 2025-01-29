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

![](https://cdn.mathpix.com/snip/images/hR5FO41bYihQ9FLBBlt4HE7tMK6lDtrgHRUuOUCriX0.original.fullsize.png)



- 기존 Stable Diffusion LDM에서 컨디셔닝을 어떻게 처리하는지 살펴보겠음.
- 먼저, 가장 널리 알려진 방법은 텍스트 프롬프트를 통한 Text-to-Image 생성
- Image-to-Image 변환 방식도 존재하지만, 사용자의 의도를 정확히 반영하는 것은 여전히 어려운 과제로 남음
- 이러한 구체적인 요구사항을 반영하고 제어할 수 있도록 해주는 것이 ControlNet이고 이는 간단하면서도 근본적인 문제를 해결할 수 있는 모델이라고 필자는 설명



![](https://cdn.mathpix.com/snip/images/VoYPN658c6bfbc-MWSrbW9Y-vBzRtfQOaa-LmB7KdLc.original.fullsize.png)

-  이 모델은 한마디로 위 그림과 같이 x의 특성(feature)을 입력받아 y라는 출력값을 생성할 때, ControlNet 모듈의 조건(condition) c를 추가로 고려하여 최종 y 값을 도출한다고 보면 된다. 


![](https://cdn.mathpix.com/snip/images/2iNLaVfzFVsBGPqYS-NgQp6GYM7n2VpgYNNjT4v2qAU.original.fullsize.png)


- 위 이미지를 좀 더 구체적으로 이해해보면 "pretrained large diffusion models"을 가져와 활용
- 학습 시에는 기존 large 모델의 장점을 최대한 살리기 위해 frozen 상태로 유지하고,
- 추가적인 condition c를 접목하여 다른 모델과 함께 학습시키는 방식으로 기존 대형 모델의 장점을 극대화함


여기서 가장 중용한 Network를 기준으로, 아래 2개의 개념을 이해보면 된다. 

1. zero-convolution
2. trainable copy

자 각각 무엇인지 한번 살펴보자.

### 1.  zero-convolution : 
- Zero Convolution Layer는 1×1 컨볼루션 레이어로서, weight와 bias를 모두 0으로 초기화한다. 이렇게 하면 초기 단계에서는 출력에 영향을 주지 않다가, 학습을 통해 점차 필요한 파라미터 값이 최적화되어 원하는 정보를 학습할 수 있게 된다. 이렇게 하는 이유는  아래 이미지를 참고하면 된다


![](https://cdn.mathpix.com/snip/images/48hMZkKzpNTRI-p5QMDH1s6gPNbupxDRMJr0a940aqE.original.fullsize.png)


![](https://cdn.mathpix.com/snip/images/AELbV6cfab3P5IsRuAgEwCEcWc5o17VRhw3qpQHKogM.original.fullsize.png)


위 내용을 요약하자면 
- 처음에는 노이즈로 초기화해 컨디션 입력(c)이 초기부터 큰 영향을 주지 않도록 하고 학습 초반에 trainable copy(학습 가능한 복사본)가 망가지는 것을 막고, 학습이 올바른 방향으로 진행될 수 있게 하고 점차적으로 최적화 과정을 통해 성능을 끌어올리기 위해서다 라고 설명하고 있다. 




## 다음으로 식을 이해해 보자, 


![](https://cdn.mathpix.com/snip/images/17_LDq4pTZ2fpfQ9pDBy_g9RkQBLTXSP_ovGyEN1Fi4.original.fullsize.png)



$$ 
\begin{aligned}&y=w x+b\\&\partial y / \partial w=x, \partial y / \partial x=w, \partial y / \partial b=1\\&\text { if } w=0 \text { and } x \neq 0\\& \partial y / \partial w \neq 0, \partial y / \partial x=0, \partial y / \partial b \neq 0\end{aligned}
$$

 
- y는 weight *x+bias로 계산된다.
- 학습(Backpropagation) 시, weight와 bias 각각에 대해 편미분을 구해 업데이트한다.
- 비록 초기 weight값이 0이라도, 입력 x가 0이 아니라면 역전파 과정을 통해 weight가 0이 아닌 값으로 학습될 수 있다.
- 위에 식이 이러한 과정을 수식으로 보여준다.





![](https://cdn.mathpix.com/snip/images/QTON_AlO7DZSwYEf-jhaOc1sX-6WmXrA4qCdjB2TwAs.original.fullsize.png )




$$
y_c=\mathcal{F}(\boldsymbol{x} ; \Theta)+\mathcal{Z}\left(\mathcal{F}\left(\boldsymbol{x}+\mathcal{Z}\left(\boldsymbol{c} ; \Theta_{21}\right) ; \Theta_c\right) ; \Theta_{22}\right)
$$

- Z 함수는 ControlNet을 거쳐서 나온값이라 보면 되고 

## First Training Step 


### i) Initialised to zero
$$
\left\{\begin{array}{l}
\mathcal{Z}\left(c ; \Theta_{\mathrm{z} 1}\right)=0 \\
\mathcal{F}\left(x+\mathcal{Z}\left(c ; \Theta_{\mathrm{z1}}\right) ; \Theta_c\right)=\mathcal{F}\left(x ; \Theta_c\right)=\mathcal{F}(x ; \Theta) \\
\mathcal{Z}\left(\mathcal{F}\left(x+\mathcal{Z}\left(\boldsymbol{c} ; \Theta_{\mathrm{z} 1}\right) ; \Theta_{\mathrm{c}}\right) ; \Theta_{z 2}\right)=\mathcal{Z}\left(\mathcal{F}\left(x ; \Theta_{\mathrm{c}}\right) ; \Theta_{\mathrm{z2}}\right)=\mathbf{0}
\end{array}\right.
$$


### ii)  Forward pass:

$$
Z(I ;\{W, B\})_{p, i}=B_i+\sum_{j=1}^c I_{p, j} W_{i, j}
$$


$$

\begin{aligned}
&\text { (2) Gradient w.r.t. bias } B_i \text { : }\\
&\frac{\partial \mathcal{L}}{\partial B_i}=\sum_p \frac{\partial \mathcal{L}}{\partial Z_{p, i}},
\end{aligned}

$$

$$
\begin{aligned}
&\text { (3) Gradient w.r.t. weight } W_{i, j} \text { : }\\
&\frac{\partial \mathcal{L}}{\partial W_{i, j}}=\sum_p\left(\frac{\partial \mathcal{L}}{\partial Z_{p, i}} I_{p, j}\right) .
\end{aligned}
$$

$$
\begin{aligned}
&\text { (4) Gradient w.r.t. input } I_{p, j} \text { : }\\
&\frac{\partial \mathcal{L}}{\partial I_{p, j}}=\sum_i\left(\frac{\partial \mathcal{L}}{\partial Z_{p, i}} W_{i, j}\right) .
\end{aligned}
$$

### iii) Proof of not equal zero

$$
\mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}=\boldsymbol{B}_i+\sum_j^c \boldsymbol{I}_{p, i} \boldsymbol{W}_{i, j} \quad\left\{\begin{array}{l}
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{B}_i}=1 \\
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{I}_{p, i}}=\sum_j^c \boldsymbol{W}_{i, j}=0 \\
\frac{\partial \mathcal{Z}(\boldsymbol{I} ;\{\boldsymbol{W}, \boldsymbol{B}\})_{p, i}}{\partial \boldsymbol{W}_{i, j}}=\boldsymbol{I}_{p, i} \neq \mathbf{0}
\end{array}\right.
$$




### v) Optimization
$$
W^*=W-\beta_{\mathrm{lr}} \cdot \frac{\partial \mathcal{L}}{\partial \mathcal{Z}(I ;\{W, B\})} \odot \frac{\partial \mathcal{Z}(I ;\{W, B\})}{\partial W} \neq 0
$$



위 식을 가단하게 정리하면 아래와 같다 

- **초기화**: Zero Convolution Layer는 가중치(weight)와 바이어스(bias)를 모두 0으로 초기화한다.
- **첫 번째 학습 스텝**: 초기 가중치가 0이므로, 처음에는 출력에 영향을 주지 못한다(출력이 거의 0에 가까움).
- **역전파 과정**: 학습 과정에서 가중치가 0이더라도, 입력값(x)이 0이 아닌 경우엔 편미분 값이 0이 아니게 된다.
- **업데이트**: 따라서 다음 스텝에서 weight는 0이 아닌 값으로 업데이트되어, 점차 학습이 진행된다.




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

