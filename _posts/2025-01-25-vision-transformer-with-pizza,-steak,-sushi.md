---
layout: page
title: "Vision Transformer with Pizza, Steak, Sushi"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---
# Vision Transformer (ViT) with Pizza, Steak, Sushi Dataset

이 노트북은 **Vision Transformer(ViT)** 논문 구조를 간략히 구현하고, `pizza_steak_sushi` 데이터셋을 이용해 분류 문제를 학습합니다.

## 진행 순서

1. **환경 설정** 및 필요한 함수 다운로드
2. **데이터 다운로드** (`pizza_steak_sushi`)
3. **Dataset & DataLoader** 만들기
4. **ViT 모델 구현** (Patch Embedding, MSA, MLP, Transformer Encoders)
5. **학습 루프** 정의
6. **학습 실행** 및 결과 확인

```python
%%capture
# Colab 환경에서 torch, torchvision이 없다면 설치 (대부분은 기본 포함)
!pip install torch torchvision

# helper_functions.py 다운로드 (여기에 download_data 함수가 들어있음)
try:
    from helper_functions import download_data
except ImportError:
    !wget https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py
    from helper_functions import download_data

# 위 명령 결과 출력이 길 수 있으니 %%capture로 숨김.
# 실행이 끝나면 아래 셀로 넘어가세요.
```

## 1. 환경 설정

아래 셀을 통해 **PyTorch**와 **TorchVision**이 설치되었는지 확인하고, GPU를 사용 가능한지 체크합니다.

```python
import torch
import torchvision

print("[INFO] PyTorch version:", torch.__version__)
print("[INFO] TorchVision version:", torchvision.__version__)

# device 설정 (GPU가 사용 가능하면 GPU, 아니면 CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Device:", device)
```

    [INFO] PyTorch version: 2.5.1+cpu
    [INFO] TorchVision version: 0.20.1+cpu
    [INFO] Device: cpu

## 2. 데이터 다운로드

[**pizza_steak_sushi**](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip) 데이터셋을 **download_data()** 함수를 사용해 내려받습니다.

압축이 풀리면 구조는 다음과 같습니다:

```
pizza_steak_sushi/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
└── test/
    ├── pizza/
    ├── steak/
    └── sushi/
```

```python
from helper_functions import download_data
image_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"  # 이 폴더명으로 다운로드 & 압축해제
)
print("[INFO] Data downloaded to:", image_path)
```

    [INFO] Did not find data/pizza_steak_sushi directory, creating one...
    [INFO] Downloading pizza_steak_sushi.zip from https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip...
    [INFO] Unzipping pizza_steak_sushi.zip data...
    [INFO] Data downloaded to: data/pizza_steak_sushi

## 3. 데이터셋 / 데이터로더 생성

파이토치의 [**ImageFolder**](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) 클래스를 사용해 이미지 폴더 구조를 **Dataset**으로 만들고, 이를 **DataLoader**로 감싸 학습에 활용합니다.

- **변환(Transform)**: 이미지를 $(224, 224)$ 사이즈로 리사이즈, 텐서 형태로 변환
- **배치 크기**: 32
- **shuffle=True** (학습용)

```python
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 이미지 변환 정의
#  - Resize(224,224): 이미지를 224x224로 맞춤
#  - ToTensor(): PyTorch 텐서([C,H,W])로 변환
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 학습/테스트 경로 설정
train_dir = image_path / "train"
test_dir = image_path / "test"

# ImageFolder를 이용해 Dataset 만들기
train_dataset = ImageFolder(root=train_dir,
                            transform=manual_transforms)
test_dataset = ImageFolder(root=test_dir,
                           transform=manual_transforms)

# Dataloader로 감싸기
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False)

# 클래스 이름(폴더명) 확인
class_names = train_dataset.classes  # 예: ['pizza', 'steak', 'sushi']
print("[INFO] Class Names:", class_names)
print(f"[INFO] Total classes: {len(class_names)}")
```

    [INFO] Class Names: ['pizza', 'steak', 'sushi']
    [INFO] Total classes: 3

## 4. Vision Transformer 구현

아래 단계로 **ViT**를 구성합니다:

1. **Patch Embedding**:
   - 입력 이미지 텐서: $[B, C, H, W]$
   - $H, W$가 $\text{patch_size}$로 나누어떨어진다고 가정
   - Conv2d(kernel_size=stride=$\text{patch_size}$)로 $(H/\text{patch_size}) \times (W/\text{patch_size})$ 패치를 임베딩 벡터($\text{embedding_dim}$)로 변환
   - 최종 shape: $[B, N, D]$ (여기서 $N$은 패치 총 개수, $D$는 임베딩 차원)

2. **Class Token / Position Embedding**:
   - Class 토큰(class token)이라는 학습 가능한 벡터를 맨 앞에 붙임
   - 위치 임베딩(position embedding)도 더해줌
   - 최종 shape: $[B, N+1, D]$

3. **TransformerEncoderBlock** (Multi-Head Self Attention + MLP):
   - MultiheadSelfAttentionBlock
   - MLPBlock
   - 각 블록 사이에 residual 연결

4. **최종 분류**:
   - Transformer 출력 중 **첫 번째 토큰(class token)**만 꺼내서 Linear로 분류

이 과정을 PyTorch 코드로 하나씩 나눠서 보겠습니다.

```python
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    (1) 이미지(크기 HxW)를 (patch_size x patch_size) 크기로 나눈 뒤,
    각 패치를 embedding_dim 채널로 변환해주는 레이어.
    """
    def __init__(
        self,
        in_channels: int = 3,      # 입력 채널 수(RGB=3)
        patch_size: int = 16,     # 패치 한 변의 크기
        embedding_dim: int = 768  # 패치 임베딩 후 차원
    ):
        super().__init__()

        # Conv2d를 이용해서 (patch_size x patch_size) 단위로 이미지를 쪼개면서
        # out_channels=embedding_dim 형태로 만듦.
        # kernel_size=stride=patch_size 이므로, 가로/세로 방향으로 패치가 딱 나누어짐.
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # 2D (h_patch, w_patch)를 1D로 펼쳐서 [batch, embedding_dim, num_patches] 형태로 만들기
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch, in_channels, height, width]
        # Conv2d 적용 -> [batch, embedding_dim, h_patch, w_patch]
        x = self.patcher(x)

        # Flatten -> [batch, embedding_dim, num_patches]
        x = self.flatten(x)

        # permute -> [batch, num_patches, embedding_dim]
        x = x.permute(0, 2, 1)
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    """
    (2) Multi-Head Self Attention + LayerNorm
    - residual은 TransformerEncoderBlock에서
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        # LayerNorm: 마지막 차원(embedding_dim)에 대해 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # MultiheadAttention: query, key, value가 전부 같음(self-attention)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True  # 입력 shape: (batch, seq_len, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm
        x_normed = self.layer_norm(x)

        # Self-Attention 수행 (Q=K=V=x_normed)
        attn_output, _ = self.multihead_attn(
            x_normed,
            x_normed,
            x_normed,
            need_weights=False  # 어텐션 가중치 반환 안 함
        )
        return attn_output

class MLPBlock(nn.Module):
    """
    (3) MLP + LayerNorm
    - Linear -> GELU -> Dropout -> Linear -> Dropout
    - residual은 TransformerEncoderBlock에서
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        # LayerNorm
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # MLP 구조
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm
        x_normed = self.layer_norm(x)
        # MLP 통과
        x_out = self.mlp(x_normed)
        return x_out

class TransformerEncoderBlock(nn.Module):
    """
    (4) Transformer 블록 하나
       - MSA 블록 -> residual
       - MLP 블록 -> residual
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0.0
    ):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (A) MSA 결과 + residual
        x = self.msa_block(x) + x
        # (B) MLP 결과 + residual
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):
    """
    (5) 최종 Vision Transformer
        - PatchEmbedding
        - class 토큰, position 임베딩 추가
        - 여러 TransformerEncoderBlock 쌓기
        - 첫 번째 토큰(class token)만 꺼내서 최종 Linear 분류
    """
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 3  # pizza, steak, sushi (3개)
    ):
        super().__init__()

        # 패치 개수: (224 x 224) / (16 x 16) = 196
        self.num_patches = (img_size * img_size) // (patch_size**2)

        # class 토큰(학습 가능한 파라미터)
        # shape: (1, 1, embedding_dim)
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

        # 위치 임베딩(학습 가능), 패치 개수+1 만큼
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )

        # 패치 임베딩 후에 쓸 드롭아웃
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # (1) Patch Embedding 레이어
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        # (2) Transformer 인코더 블록을 num_transformer_layers개 쌓기
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                    attn_dropout=attn_dropout
                ) for _ in range(num_transformer_layers)
            ]
        )

        # (3) 최종 분류 레이어: LayerNorm -> Linear
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # class 토큰을 batch 크기만큼 확장: shape (batch, 1, embedding_dim)
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # (A) Patch Embedding -> [batch, num_patches, embedding_dim]
        x = self.patch_embedding(x)

        # (B) class 토큰을 맨 앞에 붙이기: [batch, num_patches+1, embedding_dim]
        x = torch.cat((class_token, x), dim=1)

        # (C) 위치 임베딩 더하기
        x = x + self.position_embedding
        # 드롭아웃 적용
        x = self.embedding_dropout(x)

        # (D) Transformer 인코더 여러 블록 통과
        x = self.transformer_encoder(x)
        # x.shape = [batch, num_patches+1, embedding_dim]

        # (E) 첫 번째 토큰(class 토큰)에 해당하는 벡터만 뽑아서 분류
        x = self.classifier(x[:, 0])
        # x.shape = [batch, num_classes]
        return x
```

## 5. 학습 루프 (Training Loop)

다음은 **기본적인 PyTorch 학습 루프** 예시입니다.

1. 모델을 `model.train()` 모드로 설정
2. Dataloader에서 배치 단위로 `(images, labels)`를 가져옴
3. 순전파(forward) → 손실(loss) 계산 → 역전파(backprop) → 가중치 업데이트
4. 에폭이 끝나면, `model.eval()` 모드로 전환하여 테스트 세트 성능 측정

위 과정을 원하는 만큼(`num_epochs`) 반복합니다.

```python
def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int = 5,
    lr: float = 1e-3
):
    """
    단순한 학습 함수:
    - num_epochs만큼 학습을 반복
    - 매 epoch마다 Train Loss/Acc, Test Loss/Acc 출력
    """
    model.to(device)  # 모델을 GPU/CPU로 이동

    # 분류 문제 -> CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # 최적화 알고리즘 -> Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # -------------------------------
        # (A) 학습 모드
        # -------------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            # 1) 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2) 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계 계산(정확도 등)
            _, preds = torch.max(outputs, dim=1)
            train_loss += loss.item() * images.size(0)
            train_correct += torch.sum(preds == labels).item()
            train_total += images.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # -------------------------------
        # (B) 평가(테스트) 모드
        # -------------------------------
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.inference_mode():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, dim=1)
                test_loss += loss.item() * images.size(0)
                test_correct += torch.sum(preds == labels).item()
                test_total += images.size(0)

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total

        # 결과 출력
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f} | "
              f"Test Loss: {epoch_test_loss:.4f}, "
              f"Test Acc: {epoch_test_acc:.4f}")
```

## 6. 모델 생성 및 학습 실행

마지막으로, **ViT 모델**을 생성하고, 앞서 만든 `train_model()` 함수로 학습을 수행합니다.

- 기본 파라미터: $\text{num_transformer_layers} = 12$, $\text{embedding_dim} = 768$, 등
- 에폭 수(`num_epochs`)과 학습률(`lr`)은 자유롭게 조정하세요.

```python
# 클래스 개수는 pizza, steak, sushi => 3개
num_classes = len(class_names)

# ViT 모델 생성
model = ViT(
    num_classes=num_classes  # 분류 대상 = 3
)

# 학습 실행 (5에폭, lr=1e-3)
train_model(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    num_epochs=5,
    lr=1e-3
)
```

    Epoch [1/5] Train Loss: 3.2139, Train Acc: 0.3111 | Test Loss: 1.3107, Test Acc: 0.3333
    Epoch [2/5] Train Loss: 1.3715, Train Acc: 0.2978 | Test Loss: 1.1559, Test Acc: 0.2533
    Epoch [3/5] Train Loss: 1.1686, Train Acc: 0.3333 | Test Loss: 1.1145, Test Acc: 0.3333
    Epoch [4/5] Train Loss: 1.1448, Train Acc: 0.3067 | Test Loss: 1.1418, Test Acc: 0.2533
    Epoch [5/5] Train Loss: 1.1668, Train Acc: 0.3333 | Test Loss: 1.1105, Test Acc: 0.4133
