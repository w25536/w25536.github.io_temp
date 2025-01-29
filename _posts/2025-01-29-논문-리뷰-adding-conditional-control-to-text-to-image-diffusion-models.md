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

**[1. Simple Introduction](#1)**

**[2. Background Knowledge: Diffusion](https://kyujinpy.tistory.com/128#background)**

**[3. Method](https://kyujinpy.tistory.com/128#method)**

**[4. Result](https://kyujinpy.tistory.com/128#result)**


---

## Introduction[#1]

![](https://blog.kakaocdn.net/dn/ln8IQ/btsEIQRkTSQ/Yy55WaOa3nbel7NPf7Dvuk/img.png)



****


ControlNet은 정말 간단한 적용만으로도, 쉽게 diffusion network를 통제할 수 있도록 설계했다.

바로 위의 그림처럼, network를 기준으로, 아래 2개의 개념을 적용한다.

**1. zero-convolution**

**2. trainable copy**

각각이 무엇인지 한번 살펴보자.


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



![](https://cdn.mathpix.com/snip/images/QTON_AlO7DZSwYEf-jhaOc1sX-6WmXrA4qCdjB2TwAs.original.fullsize.png )





$$ \begin{aligned}&y=w x+b\\&\partial y / \partial w=x, \partial y / \partial x=w, \partial y / \partial b=1\\&\text { if } w=0 \text { and } x \neq 0\\& \partial y / \partial w \neq 0, \partial y / \partial x=0, \partial y / \partial b \neq 0\end{aligned}$$


$(".latex").latex();

```latex
<div class="latex">  
	\begin{aligned}&y=w x+b\\&\partial y / \partial w=x, \partial y / \partial x=w, \partial y / \partial b=1\\&\text { if } w=0 \text { and } x \neq 0\\& \partial y / \partial w \neq 0, \partial y / \partial x=0, \partial y / \partial b \neq 0\end{aligned}  
</div>
```



<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
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
