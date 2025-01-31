---
layout: page
title: "텐서플로 함수형 (Functional) API 모델링"
description: "텐서플로 함수형 (Functional) API 모델링에 대해 알아보겠습니다."
headline: "텐서플로 함수형 (Functional) API 모델링에 대해 알아보겠습니다."
categories: tensorflow
tags: [python, tensorflow, functional api, 함수형 api, 모델링, 텐서플로우, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2021-01-07
---

본 튜토리얼은 TensorFlow Advanced Technique 시리즈의 일부 내용을 발췌하였습니다. 

이번 튜토리얼에서는 TensorFlow Functional API를 활용한 모델링 방법에 대하여 공유 드리겠습니다. Functional API는 Sequential API 활용시 순차적이며 단순한 모델 밖에 구현할 수 없다는 한계가 존재 하는데, Functional API로 구현시 multi-input/output 그리고 스킵 연결 등 보다 복잡한 모델을 구현할 수 있습니다.

<body>
<div class="border-box-sizing" id="notebook" >
<div class="container" id="notebook-container">
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="고급-모델링-기법-(Functional-API)">고급 모델링 기법 (Functional API)</h1><p>본 튜토리얼에서는 고급 텐서플로우 모델링에 대하여 배웁니다.</p>
<ul>
<li>Functional API</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="필요한-모듈-import">필요한 모듈 import</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="실습에-필요한-Dataset-로드">실습에 필요한 Dataset 로드</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">mnist</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">mnist</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">),</span> <span class="p">(</span><span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span> <span class="o">=</span> <span class="n">mnist</span><span class="o">.</span><span class="n">load_data</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x_train</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">x_test</span><span class="o">.</span><span class="n">shape</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>((60000, 28, 28), (10000, 28, 28))</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">y_train</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">y_test</span><span class="o">.</span><span class="n">shape</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>((60000,), (10000,))</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>이미지 정규화 (Normalization)</strong></p>
<ul>
<li>모든 이미지 픽셀(pixel)값들을 0~1 사이의 값으로 정규화 해 줍니다.</li>
<li>x_train, x_valid 에 대해서만 정규화합니다.</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x_train</span> <span class="o">=</span> <span class="n">x_train</span> <span class="o">/</span> <span class="mf">255.0</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">x_test</span> <span class="o">/</span> <span class="mf">255.0</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x_train</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">x_train</span><span class="o">.</span><span class="n">max</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>(0.0, 1.0)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Sequential-API">Sequential API</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><code>tf.keras.Sequential()</code>가 <code>tf.keras.models.Sequential()</code>을 레퍼런스 하고 있기 때문에 같은 객체가 생성됩니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">seq_model</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">models</span><span class="o">.</span><span class="n">Sequential</span><span class="p">()</span>
<span class="n">seq_model</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>&lt;tensorflow.python.keras.engine.sequential.Sequential at 0x7fa2e835a4a8&gt;</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">seq_model</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">Sequential</span><span class="p">()</span>
<span class="n">seq_model</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>&lt;tensorflow.python.keras.engine.sequential.Sequential at 0x7fa2e835ad68&gt;</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Sequential API를 활용하여 모델</strong>을 구축합니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">seq_model</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">Sequential</span><span class="p">([</span>
    <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Flatten</span><span class="p">(</span><span class="n">input_shape</span><span class="o">=</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span> <span class="mi">28</span><span class="p">)),</span>
    <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">256</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">),</span>
    <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">),</span>
    <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">),</span>
    <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'softmax'</span><span class="p">),</span>
<span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>모델의 <strong>요약본(summary)</strong>을 확인합니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">seq_model</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 242,762
Trainable params: 242,762
Non-trainable params: 0
_________________________________________________________________
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Functional-API">Functional API</h2><ul>
<li>이번에는 함수형 API를 활용하여 모델 구축을 진행합니다.
Functional API의 핵심은 input 인자로 이전 layer의 output을 대입하는 것입니다.</li>
<li>Functional API의 첫번째 layer는 <strong>Input layer</strong>로 이어야 합니다.</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Input-layer">Input layer</h3><ul>
<li><p>Sequential API로 모델을 구축할 때는 input_shape로 지정했지만, Functional API는 <strong>Input layer를 선언</strong>해야 합니다.</p>
</li>
<li><p>layer마다 name을 지정해 줄 수 있습니다. 지정한 name은 모델의 요약(summary) 확인시 출력됩니다.</p>
</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">input_layer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">Input</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span> <span class="mi">28</span><span class="p">),</span> <span class="n">name</span><span class="o">=</span><span class="s1">'InputLayer'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>그 다음 단계 부터는 layer를 차례대로 쌓아 주면 됩니다.</p>
<p>이전 단계의 layer output이 다음 layer의 input 값으로 들어갈 수 있도록 <strong>chain처럼 연결</strong>해 줍니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Chain-처럼-모델-쌓기">Chain 처럼 모델 쌓기</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Flatten</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="s1">'Flatten'</span><span class="p">)(</span><span class="n">input_layer</span><span class="p">)</span>
<span class="n">x2</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">256</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'Dense1'</span><span class="p">)(</span><span class="n">x1</span><span class="p">)</span>
<span class="n">x3</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'Dense2'</span><span class="p">)(</span><span class="n">x2</span><span class="p">)</span>
<span class="n">x4</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'Dense3'</span><span class="p">)(</span><span class="n">x3</span><span class="p">)</span>
<span class="n">x5</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'softmax'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'OutputLayer'</span><span class="p">)(</span><span class="n">x4</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="각-변수에-담은-layer-확인">각 변수에 담은 layer 확인</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>위의 Functional API로 모델링을 구축했다면, 각 변수에 담은 layer를 출력해 볼 수 있습니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x1</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>&lt;tf.Tensor 'Flatten/Reshape_1:0' shape=(None, 784) dtype=float32&gt;</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">x2</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>&lt;tf.Tensor 'Dense1/Relu:0' shape=(None, 256) dtype=float32&gt;</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Model-로-빌드하기">Model 로 빌드하기</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">func_model</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">Model</span><span class="p">(</span><span class="n">inputs</span><span class="o">=</span><span class="n">input_layer</span><span class="p">,</span> <span class="n">outputs</span><span class="o">=</span><span class="n">x5</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'FunctionalModel'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>model의 <code>summary()</code>로 학습해야할 parameter의 갯수가 Sequential API로 쌓은 모델과 동일한지 확인합니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">func_model</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Model: "FunctionalModel"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
InputLayer (InputLayer)      [(None, 28, 28)]          0         
_________________________________________________________________
Flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
Dense1 (Dense)               (None, 256)               200960    
_________________________________________________________________
Dense2 (Dense)               (None, 128)               32896     
_________________________________________________________________
Dense3 (Dense)               (None, 64)                8256      
_________________________________________________________________
OutputLayer (Dense)          (None, 10)                650       
=================================================================
Total params: 242,762
Trainable params: 242,762
Non-trainable params: 0
_________________________________________________________________
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Plot-Model로-시각화">Plot Model로 시각화</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><code>plot_model</code>은 빌드한 모델에 대한 시각화를 제공합니다.</p>
<p>나중에는 복잡한 모델을 코드로 구현하게 될 때, 반복문이나 조건문을 사용하기도 합니다. 그러한 경우에는 모델의 구조 파악이 매우 까다롭습니다. 하지만, <code>plot_model</code>로 <strong>시각화를 해보면 쉽게 모델의 구조를 파악</strong>해 볼 수 있습니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">tensorflow.keras.utils</span> <span class="k">import</span> <span class="n">plot_model</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">seq_model</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_png output_subarea output_execute_result">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARMAAAIjCAYAAAAgO/+IAAAABmJLR0QA/wD/AP+gvaeTAAAgAElEQVR4nO3dfVAUZ54H8G/DDMOrw8siJooaXd1YOYInYVeMFCIrasSMsiDiC0aj8dTERM9N1nMv5RlrEyuJuWzFnGu8Wy97XglaJdFospKKpkoYKomiRjeowTWHIiziwoK8DvzujxSzmfSgAz7ODMP3U9V/zNPPdP/6sfk63T3TrYmIgIjo3uz383QFROQbGCZEpATDhIiUYJgQkRKGHzZYrVZs377dE7UQUT+xf/9+XZvuk0llZSUOHDjgloKIqH+5du1aj/mg+2TSzVnyENHAVlBQgJycHKfzeM6EiJRgmBCREgwTIlKCYUJESjBMiEgJhgkRKcEwISIlGCZEpATDhIiUYJgQkRIMEyJSgmFCREowTIhICaVh8u2332Lp0qUYPnw4AgICoGmafdq6davKVVE/ERoa6rAfaJqGN954w9Nl9ZmvbY9KysKktrYWEydOxOnTp1FQUID6+nqICKxWq6pV9GtNTU0YM2YMMjIyPF2KWzU1NaGsrAwAYLFYICLYsGGDh6vqO1/bHpWUhcnu3btRXV2Nt956CxMnTkRwcLCqRQP47n+EyZMn93m+p4kIurq60NXV5elS7srbx/J+G+jb31c93hypt7766isAQFxcnKpF+pSwsDBUVFR4ugyi+0bZJ5Pm5mYA3/3RENHAc89hUlhYCE3T8MEHHwAAgoKCoGnaXT8m2mw25OfnY9q0aRgyZAiCgoIQFxeHt99+2+FQ4I033oCmabh9+zaKi4vtJ70MBoNL87vV1tZi7dq1GDlyJAICAhAdHY3MzEycOXNGty3d09WrV5GTk4Pw8HBERUUhIyOjT58ufrjc1tbWPq2ve1s1TcOwYcPwxRdfIC0tDWFhYQgODkZqaiqKi4vt/bdu3Wrv//1/j48//tje/qMf/cjlsVZtoGy/K/t6fX297sRu90ULm83m0J6VlWVfdl/264sXL2LevHmIioqyt928efOethEAID+Qn58vTprvymKxCABpaWlxaLdarQJAXnnlFYf2w4cPCwD5zW9+I7du3ZLa2lr57W9/K35+frJhwwbd8kNCQuTxxx/vcf13ml9VVSUjRoyQmJgYOXLkiDQ2Nsr58+clJSVFAgMDpaSkxOm2WCwWKSkpkaamJikqKpKgoCBJTEx0dUh0ehqj3q4vPj5eQkJCJCkpyd7/iy++kEcffVQCAgLkxIkTLo1NQkKCREVF6drvNtapqakSGRkpVqvVpe0uKyuzb58z/W3777Y9P9SbfX369Oni5+cn33zzjW45SUlJsnfvXvvrvu7XKSkpcvz4cbl9+7aUlpaKv7+/1NbWurQtd8iHAo+GyZQpU3TLWbRokRiNRmloaHBov5cwWbJkiQBw+IcQEblx44aYTCZJSEhwui2HDx92aM/KyhIALg/8D90tTFxdX3x8vACQsrIyh/Zz584JAImPj3doV/3HlJKSIhEREbqdtSeuhkl/2f6+hImr+/of//hHASCrV6926Hvy5EkZOnSotLe329v6ul8fPXrUpbqduVOYeOxLaxkZGTh+/LiuPT4+Hh0dHbhw4YKydRUWFsLPz093WXbIkCF45JFHcOrUKVy7dk33vsTERIfXsbGxAICqqipltfV1fSEhIRg/frxDW1xcHB588EGcPXsWN27cuC81AsCJEydw69YtJCUlKV1uf9n+3urNvp6eno64uDjs2bMHdXV19vbXX38dzz33HIxGo72tr/v1T3/6UxWbpeOxMGloaMDLL7+MuLg4RERE2I/dfvnLXwL4+wnde9XW1oaGhgZ0dXXBbDbrjktPnz4NALh8+bLuvWaz2eF1QEAAANy3y7u9WV94eLjTZQwePBgA8Je//EVxdfefr25/b/f1F154Ac3NzXj33XcBAJcuXcKnn36KZ555xt7nXvbrkJCQ+7KdHguT2bNn45VXXsGKFStw6dIldHV1QUTw1ltvAfjuexnfp2naHZfX03yTyYTw8HAYDAZ0dHRARJxOqampajbMTerq6nRjBPz9j6j7jwoA/Pz80N7erutbX1/vdNl3G2tv0J+2v7f7+sKFCxETE4N33nkHbW1tePPNN7FkyRJERETY+3jjfu2RMOns7ERxcTGGDBmCtWvXIjo62v4P2NLS4vQ9wcHBDjvET37yE+zatcul+ZmZmbDZbA5n+rtt27YNw4cPh81mU7Jt7tLa2oovvvjCoe2rr75CVVUV4uPj8cADD9jbH3jgAVy/ft2hb3V1Nf7v//7P6bLvNtbewNu332AwoLy8vE/7uslkwurVq/GXv/wFb775Jvbu3Yvnn39e18/b9muPhIm/vz+mTJmC6upqvP7667h5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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>주요 parameter</strong></p>
<ul>
<li>show_shapes: 모델의 shape 출력</li>
<li>show_layer_names: 모델의 name 출력</li>
<li>to_file: 저장할 파일의 이름</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">func_model</span><span class="p">,</span> <span class="n">show_shapes</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">show_layer_names</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">to_file</span><span class="o">=</span><span class="s1">'model.png'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">

<div class="output_png output_subarea output_execute_result">
<img src="data:image/png;base64,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T0/H6tWrUVpaiuPHj8POzg6TJ0/GqVOnmtz3448/RkZGRpPb58yZg8TERLz11lutipHujUmYiJplZ2enLaDRHcc3xtWrVzFp0iQ888wz2opfGsnJyQgNDYVSqUTfvn3xr3/9C56enkhOTkZpaanJYy5atAg///wzvvrqKzz22GNQKBTo06cP0tPTte/wG1JYWIj4+HjMnDmzyT4DBgzAnj17kJSUxNXbzIhJmIioDaxatQrFxcVYunSpTvvevXvx3HPP6bTJ5XIMGjQIN2/e1FsG1VglJSXYuHEjZsyYAVdXV51tSqUStbW1GDJkiMF958yZg8jISAQHB99zDLVajYiICCxcuJDvupsJkzARUSsJIZCWlgZ/f394eHgYtU9eXh4AoGfPniaNqSmp2dIZgs2bN+PUqVNITk42qn9YWBjy8/N1FqihtsMkTNRFlJWV4dVXX8WAAQMgl8vh5OSEJ598Et9++622z4oVK7Q1le/8471//35tu6aABXBrGlUmk6G6uhpHjhzR9tEsPqLZLpPJ4OXlhaysLIwfPx729vawtbXFuHHjdFYSa+vxO4qcnByUlJRArVYb1X/Lli3Izc2Ft7c3Bg0aZNKYP/74IwDAyckJCxcuRO/evSGXy3H//fcjNjbW4FKy+fn5WLhwITZv3gx7e3ujxhk+fDgA4MsvvzQpTro3JmGiLqC4uBijRo3C9u3bkZKSgitXruDYsWOwtbXF+PHjkZaWBgB48803IYSAUqnU2f+JJ56AEAK+vr467QkJCdr+jz76qHbt3zuXNBVCQK1Wo6KiAnFxcVixYgWKi4vx3//+F+Xl5QgKCsJ3331nlvE1WrtCW2udPHkSAJot0nH69GnExcVh9uzZcHJywrZt23RWgGsJTVWx2bNno6SkBN999x1KS0uxfPlybN68GQEBAdriIhoxMTGYPn26zopvzfH09ARw+zNS22ISJuoCEhMTceHCBaxZswYTJ06Eg4MDvL29sX37dri7uyM2NhYlJSVmjaG6uhoffvghAgICoFQq4efnh23btuHGjRuIi4sz69iNjY06RRnamyYh3quSFgAMGzYMu3btwvz583Hy5MlWrbleW1sLAFAoFEhPT0f//v3h6OiIWbNmITExEefOncMHH3yg7Z+amorz589j1apVLRrHwcEBMplM+xmpbTEJE3UBmndOQ0NDddpt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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Sequential-&amp;-Functional-API-모델의-학습">Sequential &amp; Functional API 모델의 학습</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>모델의 학습 단계는 동일합니다.</p>
<p>나중에는 custom 학습 방법도 알아야 하지만, 지금은 <strong>compile(), fit() 으로 학습하고, evaluate()로 모델을 평가</strong>하도록 하겠습니다.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Sequential-API-모델의-학습">Sequential API 모델의 학습</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">seq_model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">optimizers</span><span class="o">.</span><span class="n">Adam</span><span class="p">(),</span>
              <span class="n">loss</span><span class="o">=</span><span class="s1">'sparse_categorical_crossentropy'</span><span class="p">,</span>
              <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">'accuracy'</span><span class="p">])</span>
<span class="n">seq_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="n">seq_model</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Epoch 1/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2124 - accuracy: 0.9361
Epoch 2/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0892 - accuracy: 0.9729
Epoch 3/3
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0630 - accuracy: 0.9803
313/313 [==============================] - 1s 2ms/step - loss: 0.0800 - accuracy: 0.9758
</pre>
</div>
</div>
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>[0.08002541959285736, 0.9757999777793884]</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Functional-API-모델의-학습">Functional API 모델의 학습</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">func_model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">optimizers</span><span class="o">.</span><span class="n">Adam</span><span class="p">(),</span>
              <span class="n">loss</span><span class="o">=</span><span class="s1">'sparse_categorical_crossentropy'</span><span class="p">,</span>
              <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">'accuracy'</span><span class="p">])</span>
<span class="n">func_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="n">func_model</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Epoch 1/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2135 - accuracy: 0.9359
Epoch 2/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0891 - accuracy: 0.9722
Epoch 3/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0626 - accuracy: 0.9798
313/313 [==============================] - 1s 2ms/step - loss: 0.0808 - accuracy: 0.9751
</pre>
</div>
</div>
<div class="output_area">

<div class="output_text output_subarea output_execute_result">
<pre>[0.08081302791833878, 0.9750999808311462]</pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</body>