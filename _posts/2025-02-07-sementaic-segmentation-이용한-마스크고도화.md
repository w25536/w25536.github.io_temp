---
layout: page
title: "[팀프로잭트] Sementaic segmentation 이용한 Wall mask 마스크 고도화 및 추출"
description: ""
headline: ""
tags:
  - pytorch
  - 파이토치
  - 딥러닝
categories: deep-learning
comments: true
published: true
---

Semantic segmentation

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png)




### SegFormer 모델 개요

- SegFormer모델은, Transformers와 경량 MLP 디코더를 결합하여 이미지 분할에서 뛰어난 성능을 보임.
- 이 모델은 계층적 Transformer 인코더와 경량 MLP 디코더 헤드로 구성되어 있으며, ADE20K와 Cityscapes와 같은 이미지 분할벤치마크에서 좋은 결과를 달성
- SegFormer의 주요 특징은 다음과 같다
    - 다중 스케일 특징 출력: 새로운 계층 구조의 Transformer 인코더를 포함
    - 복잡한 디코더 회피: MLP 디코더는 다양한 레이어에서 정보를 집계하여 강력한 표현을 생성
- SegFormer는 SegFormer-B0에서 SegFormer-B5까지의 모델 시리즈로 확장 가능하며, 이전 모델들보다 성능과 효율성이 크게 향상됨
- ==예를 들어, SegFormer-B4는 64M 파라미터로 ADE20K에서 50.3% mIoU를 달성하며, 이전 최상의 방법보다 5배 작고 2.2% 더 나은 성능을 보임.==

![]({{site.baseurl}}/images/2025-02-07/CleanShot 2025-02-10 at 19.45.39@2x.png)

### SegFormer 모델 설명

- SegFormer는 계층적 Transformer 블록들의 인코더와 경량의 모든-MLP 디코더 헤드로 구성되어 있다.
- SegformerForSemanticSegmentation은 이미지를 분할하기 위한 모든-MLP 디코더 헤드를 추가하여 동작한다.
- SegFormer 저자들은 ImageNet-1k에서 Transformer 인코더를 사전 학습한 후, 분류 헤드를 제거하고 ADE20K, Cityscapes, COCO-stuff 데이터셋으로 모델을 미세 조정
- SegFormer는 입력 크기에 상관없이 동작하며, 입력을 config.patch_sizes로 나누어 떨어지도록 패딩 처리한다.
- SegformerImageProcessor를 통해 이미지와 분할 맵을 준비할 수 있으나, 원본 논문에서 사용된 모든 데이터 증강 방법을 포함하지 않으므로, ADE20K 데이터셋의 경우 무작위 자르기와 패딩을 통한 정규화가 중요하다.

### 요약

-  **효율적 의미 분할 프레임워크**: Transformer와 경량 MLP 디코더를 결합하여 구현
- **계층적 Transformer 인코더**: 다중 스케일 특징을 출력하며, 위치 인코딩 없이 성능 저하를 방지
- **경량 MLP 디코더**: 복잡한 디코더 없이 로컬 및 글로벌 주의를 결합하여 강력한 표현 제공


### 모델 크기 및 성능

| **모델 변형** | **깊이**        | **숨겨진 크기**          | **디코더 숨겨진 크기** | **파라미터 (M)** | **ImageNet-1k Top 1** |
| --------- | ------------- | ------------------- | -------------- | ------------ | --------------------- |
| MiT-b0    | [2, 2, 2, 2]  | [32, 64, 160, 256]  | 256            | 3.7          | 70.5                  |
| MiT-b1    | [2, 2, 2, 2]  | [64, 128, 320, 512] | 256            | 14.0         | 78.7                  |
| MiT-b2    | [3, 4, 6, 3]  | [64, 128, 320, 512] | 768            | 25.4         | 81.6                  |
| MiT-b3    | [3, 4, 18, 3] | [64, 128, 320, 512] | 768            | 45.2         | 83.1                  |
| MiT-b4    | [3, 8, 27, 3] | [64, 128, 320, 512] | 768            | 62.6         | 83.6                  |
| MiT-b5    | [3, 6, 40, 3] | [64, 128, 320, 512] | 768            | 82.0         | 83.8                  |

- 필자는 Wall segmentation을 진행 했을때보다 훨씬더 좋은 성능이 나와 프로젝트는 이 모델을 사용하기로 하였다. 
- 데이터셋은 동등하게 150개의 서로 다른 카테고리가 라벨링된 [MIT ADE20K Scene parsing dataset](http://sceneparsing.csail.mit.edu/) 사용되었고 결과는 CNN 보다 transformer 기반이 훨씬 결과가 좋은 성능 가진다.

### 다음은 코드 리뷰를 진행 하겠다.  


```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  
```


- 필자는 공개되지 않은  SOTA SegFormer 기반으로 finet-uned 된 모델을 사용가능 b0~b4를 이용해 segmentation을 진행하도록 하겠다.


```python

# # 1) Prepare the test image
# # (Make sure to import os so that os.listdir works)
# image_files = [f for f in os.listdir('wall_test_pics') if f.endswith(('.jpg', '.jpeg', '.png'))]
# # Here we simply pick the 5th image; adjust the index as needed
# image = Image.open(os.path.join('wall_test_pics', image_files[12]))

image = Image.open('wall_test_pics/dong.jpeg')

# 2) Load the image processor and model
processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.eval()  # set model to evaluation mode

# Optional: Print the mapping between class IDs and labels
print("ID to Label Mapping:")
for k, v in model.config.id2label.items():
    print(f"{k}: {v}")

# 3) Preprocess the image to a PyTorch tensor
inputs = processor(images=image, return_tensors="pt")

# 4) Run inference (forward pass) with no gradient calculation
with torch.no_grad():
    outputs = model(**inputs)

# The model output logits have shape:
#   (batch_size, num_labels, height/4, width/4)
logits = outputs.logits
print("Logits shape:", logits.shape)

# 5) Post-process to obtain the segmentation mask
# The processor resizes the result to the original image size and applies argmax.
segmentation = processor.post_process_semantic_segmentation(
    outputs=outputs, 
    target_sizes=[image.size[::-1]]  # image.size is (width, height); reverse it to (height, width)
)[0]

print("Segmentation mask shape:", segmentation.shape)
print("Segmentation mask example (top-left 5x5):\n", segmentation[:5, :5])

# 6) Convert multi-class segmentation to binary segmentation
# In this example, we extract the "wall" class.
# You can change this target class name if desired.
target_class_name = "wall"
target_class = None

# Look up the class ID corresponding to the target class name (case-insensitive)
for k, v in model.config.id2label.items():
    if v.lower() == target_class_name.lower():
        target_class = int(k)
        break
# Create a dictionary to store class counts
class_counts = {}

# Count occurrences of each class in the segmentation mask
for class_id in segmentation.unique():
    class_id = int(class_id)  # Convert tensor to int
    class_name = model.config.id2label[class_id]
    count = (segmentation == class_id).sum().item()
    class_counts[class_name] = count

# Print detected classes and their pixel counts
print("\nDetected classes and pixel counts:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} pixels")

if target_class is None:
    raise ValueError(f"Target class '{target_class_name}' not found in the id2label mapping.")

# Create the binary mask: pixels equal to the target class are 1, others 0
binary_mask = (segmentation == target_class).numpy().astype(np.uint8)
print("Unique values in binary mask:", np.unique(binary_mask))
```

- 위 코드 `model.config.id2label.items` 에서 class 추출 코드를 추가 
- semantic segmentation을 통해  class 구별이 가능하다
- 추출된 코드에서 px 넓이를 확인 할 수 있다 


### 추출된 코드 클라스 확인  

```python
for k, v in model.config.id2label.items():
    if v.lower() == target_class_name.lower():
        target_class = int(k)
        break
```

### 추출된 target_class를 넘파이로 변환 시켜 binary mask 생성 
```python 
binary_mask = (segmentation == target_class).numpy().astype(np.uint8)
print("Unique values in binary mask:", np.unique(binary_mask))
```




![]({{site.baseurl}}/images/2025-02-07/output 1.png)


```bash
cushion: 17635 pixels 
pillow: 53337 pixels
apparel: 230 pixels
```



![]({{site.baseurl}}/images/2025-02-07/output2.png)

```bash
light: 29534 pixels 
bag: 26031 pixels 
plate: 4009 pixels
```



아래는 다양한 벽에 대한 테스팅을 진행해 보았고 다음과 같은 결과를 얻을 수 있었다. 

![]({{site.baseurl}}/images/2025-02-07/모든_segmentation_결과.png)


- 실험 결과 분석에 따르면, 일반적인 내부 벽면에 대한 인식 성능은 매우 우수 하지만 
- 벽화나 특수한 패턴이 그려진 벽면의 경우, 모델의 인식 성능이 현저히 저하되는 한계점이 발견.
- 이는 학습 데이터셋에 이러한 특수한 케이스가 충분히 포함되지 않았을 가능성을 시사,
- 추후 위같은 문제를 결하기 위해서는 다양한 벽면 패턴이 포함된 추가 데이터 수집 및 학습이 필요할 것으로 판단


### 필자는 다음과 같은 방법으로 기존에 있던 Segmentation 추가하기로 했다

- 먼저 아래의 그림을 보자 

![]({{site.baseurl}}/images/2025-02-07/output333.png)

- 아래 마스크의 벽이인식이 잘되지 않은것을 확인 할 수 있다. 


![]({{site.baseurl}}/images/2025-02-07/outpu123123t.png)

```bash

Detected classes and pixel counts: 
wall: 22045 pixels 
building: 437120 pixels 
tree: 133899 pixels 
sidewalk: 79584 pixels 
earth: 47652 pixels
```

분석 결과, 'building' 클래스가 437,120 픽셀의 넓은 영역을 차지하고 있음을 확인했다. 이를 활용하여 벽화 segmentation의 성능을 개선하기 위해 다음과 같은 접근 방식을 적용했다:

1. 기존 클래스 탐지 로직에 'building' 클래스 키워드를 추가
2. 이를 통해 벽화가 포함된 건물 영역에 대한 마스크 segmentation의 정확도가 향상
3. 결과적으로 전체적인 segmentation 성능이 개선됨

이러한 방법을 통해 벽화가 포함된 건물 영역에 대한 보다 정확한 segmentation이 가능해졌다.


![]({{site.baseurl}}/images/2025-02-07/output123132.png)

---
### 정리 

- 실험에서는 **세 가지 공개 데이터셋**, 즉 Cityscapes, ADE20K, COCO-Stuff를 사용했다. 
- ADE20K 데이터셋은 **150개의 세부적인 의미론적 개념**을 포함하여, 총 20,210장의 이미지를 포함한다.
- Cityscapes는 **5000장의 고해상도 이미지**로 구성되며, 19개의 카테고리로 주석이 달려 있다. 
- COCO-Stuff는 **172개의 레이블**로 164,000개의 이미지로 구성되며, 훈련용, 검증용, 테스트용 이미지가 포함되어 있다


![]({{site.baseurl}}/images/2025-02-07/CleanShot 2025-02-10 at 19.43.49@2x.png)


- SegFormer encoder는 **로컬** 및 **비로컬** 피처를 결합하는 것이 필요하며, 단순히 고수준 피처에만 의존해서는 안 된다.
- ADE20K와 Cityscapes 데이터셋에서 SegFormer-B0는 37.4% mIoU를 달성하며, 3.8M 파라미터와 8.4G FLOPs로 모든 실시간 대비에서 우수한 성능을 보인다.
- SegFormer-B5는 이전 최의 성능인 SETR보다 1.6% 높은 51.8% mIoU를 달성하며, 훨씬 더 효율적이다.[
- COCO-Stuff 데이터셋 평가에서는 SegFormer-B5가 46.7% mIoU를 기록하며, SETR 보다 0.9% 높은 성능을 보여준다.
- SegFormer는 **로버스트니스** 측면에서도 뛰어난 성능을 보이며, Gaussian Noise에 대해 588% 개선된 결과를 나타내고, 눈 날씨 관련 테스트에 대해서는 295% 개선을 기록했다.
- 아래 이미지를 보면 확인 할 수 있다. 

![]({{site.baseurl}}/images/2025-02-07/CleanShot 2025-02-10 at 19.40.37@2x.png)


이번 프로잭트에 SegFormer를 사용되는 이유는 다음과 같다 

- DeepLabV3+의 효과적 수용 영역(ERF)는 상대적으로 고정된 패턴을 보여주는 반면, SegFormer는 도로, 자동차, 건물의 패턴을 감지하는 데 더 민감
- **Transformer** 인코더가 **ConvNets**보다 더 강력한 **특징 추출 능력**을 가지고 있음
- **SegFormer**와 DeepLabV3+의 성능을 비교한 결과, DeepLabV3+는 노이즈 세기가 증가함에 따라 현저한 성능 저하를 나타내는 반면 , SegFormer는 상대적으로 안정적인 성능을 유지
- SegFormer는 모든 유형의 변조 및 오염에서 DeepLabV3+에 비해 상당한 이점을 보이며, 탁월한 **제로샷 강 건성(zero-shot robustness)을 입증
- SegFormer는 모든 유형의 변조 및 오염에서 DeepLabV3+에 비해 상당한 이점을 보이며, 탁월한 제로샷 강 건성(zero-shot robustness)을 입증


### 향후 실험 계획 

- FCN, DNN 기반 세그멘테이션, ResNest 등 다양한 접근법으로 마스크 개선 포인트를 분석한다.
- 벽 가림 객체 제거 시, 다른 클래스 감지되면 Lama 인페인팅 적용을 고려하며, 이를 위해 프론트엔드에서 바운딩 박스 좌표 전달 기능을 추가한다.
- 이미지 스티칭 및 기존 문제점 개선을 통해 최종 마무리한다.