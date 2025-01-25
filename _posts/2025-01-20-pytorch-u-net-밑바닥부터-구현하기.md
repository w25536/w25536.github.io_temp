---
layout: page
title: "[Pytorch] U-Net 밑바닥부터 구현하기"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle
```

```python
# Function to generate an image with shapes and labels
def create_image_with_shapes_and_labels(image_size=(256, 256)):
    """
    Creates a dummy RGB image with shapes (circle, rectangle) and corresponding labels.
    Args:
        image_size (tuple): Size of the image (H, W).
    Returns:
        torch.Tensor: RGB image tensor (3, H, W).
        torch.Tensor: Label tensor (H, W) with classes 0 (background), 1 (circle), 2 (rectangle).
    """
    image = np.zeros((*image_size, 3), dtype=np.float32)  # Blank RGB image
    label = np.zeros(image_size, dtype=np.int64)  # Label map (H, W)  0: background, 1: circle, 2: rectangle

    # Draw a circle
    rr, cc = disk((64, 64), 40) # (64, 64) 중심, 반지름 40
    image[rr, cc, 0] = 1.0  # Red circle
    label[rr, cc] = 1  # Class 1: Circle

    # Draw a rectangle
    start = (120, 120)
    extent = (50, 80)
    rr, cc = rectangle(start=start, extent=extent)
    image[rr, cc, 1] = 1.0  # Green rectangle
    label[rr, cc] = 2  # Class 2: Rectangle

    # Normalize to range [0, 1]
    image = (image - image.min()) / (image.max() - image.min()) # 0~1 사이로 정규화 
    return torch.tensor(image).permute(2, 0, 1), torch.tensor(label)  # (C, H, W), (H, W) #채널 순서 맞춤
    #(H, W, C) -> (C, H, W) 바꿔줘야 한다. 
```

```python
# Define a basic double convolution block
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3x3 커널 사이즈, 패딩 1, 바이어스 없음
        nn.BatchNorm2d(out_channels), # 배치 정규화 
        nn.ReLU(inplace=True), # 인플레이스 
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3x3 커널 사이즈, 패딩 1, 바이어스 없음 batch_norm이 적용이 된 경우 bias로 빼준다.      
        nn.BatchNorm2d(out_channels), # 배치 정규화 
        nn.ReLU(inplace=True), # 인플레이스 
    )
```

```python
# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) #upsamping 수행후 double_conv 적용
        self.dec4 = double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(kernel_size=2)(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1) 
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out_conv(dec1)
        return out
```

```python
# Training loop
def train_model(model, optimizer, criterion, num_epochs, input_image, ground_truth):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_image)
        loss = criterion(outputs, ground_truth.unsqueeze(0))  # Add batch dimension to ground_truth

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete!")
```

```python
# Visualization of results
def visualize_results(input_image, output_prediction, ground_truth=None):
    input_image = input_image.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
    output_prediction = torch.argmax(output_prediction, dim=1).squeeze().cpu().numpy()  # Convert to label map ouput prediction dim = 1 채널 원에서 결정되는거다
    if ground_truth is not None:
        ground_truth = ground_truth.cpu().numpy()

    # Plot the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(input_image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(output_prediction, cmap="jet")
    ax[1].set_title("Model Prediction")
    ax[1].axis("off")

    if ground_truth is not None:
        ax[2].imshow(ground_truth, cmap="jet")
        ax[2].set_title("Ground Truth")
        ax[2].axis("off")

    plt.tight_layout()
    plt.show()
```

```python
# Example usage
if __name__ == "__main__":
    # Create synthetic data
    input_image, ground_truth = create_image_with_shapes_and_labels()
    input_image = input_image.unsqueeze(0)  # Add batch dimension
    ground_truth = ground_truth  # (H, W)

    # Instantiate U-Net model
    num_classes = 3  # Background, Circle, Rectangle
    #model = UNet(in_channels=3, out_channels=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=num_classes).to(device)
    input_image = input_image.to(device)
    ground_truth = ground_truth.to(device)


    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, optimizer, criterion, num_epochs=10, input_image=input_image, ground_truth=ground_truth)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        output_prediction = model(input_image)

    # Visualize results
    visualize_results(input_image.squeeze(), output_prediction, ground_truth)
```

    Epoch [1/10], Loss: 1.1136
    Epoch [2/10], Loss: 0.7071
    Epoch [3/10], Loss: 0.5737
    Epoch [4/10], Loss: 0.5214
    Epoch [5/10], Loss: 0.4775
    Epoch [6/10], Loss: 0.4437
    Epoch [7/10], Loss: 0.4149
    Epoch [8/10], Loss: 0.3886
    Epoch [9/10], Loss: 0.3636
    Epoch [10/10], Loss: 0.3427
    Training complete!

![iUNet_코드의_사본_6_1.png]({{site.baseurl}}/images/2025-01-20/UNet_코드의_사본_6_1.png)

해당 U-Net은 가장 기본적인 형태를 예시로 든 것이며, 실제 적용 상황에 따라

* Encoder나 Decoder의 채널 수(64→128→256→512…)를 축소/확장,
* Dropout이나 Residual Block을 추가,
* Attention Mechanism(Attention U-Net) 또는 Dense Skip(UNet++)

등을 적용하여 성능을 높이거나 모델 크기를 조절할 수 있음.
