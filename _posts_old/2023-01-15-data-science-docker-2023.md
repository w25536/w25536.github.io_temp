---
layout: page
title: "[2023년 UPDATE] 머신러닝/딥러닝(PyTorch, TensorFlow) 최신 도커(docker) 업데이트 안내"
description: "[2023년 UPDATE] 머신러닝/딥러닝(PyTorch, TensorFlow) 최신 도커 업데이트 안내입니다."
headline: "[2023년 UPDATE] 머신러닝/딥러닝(PyTorch, TensorFlow) 최신 도커 업데이트 안내입니다."
categories: data_science
tags: [python, 파이썬, docker, 도커, 도커 허브, 데이터사이언스 도커, 딥러닝 도커, 머신러닝 도커, 도커 설치, 도커 사용법, pytorch 도커, tensorflow 도커, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-15
---

2023년 01월 15일 새해를 맞아 데이터 분석 / 머신러닝 / 딥러닝 주요 파이썬 패키지를 의존성 충돌 없이 설치, 그리고 한글 폰트, 형태소 분석기 등 한글 전처리 관련 도구가 사전에 설치된 **도커(Docker) 이미지를 리뉴얼 하여 배포** 하였습니다. 

기존의 캐글(kaggle)에서 배포하는 도커 이미지를 기반으로 확장하여 작업하였으나, 2022년 부터는 `tensorflow` 에서 유지보수하고 있는 `tensorflow/tensorflow:2.11.0-gpu-jupyter`를 확장하여 작업하였습니다. `tensorflow`, `pytorch`의 최신 라이브러리가 설치되어 있고, 뿐만 아니라 `LightGBM`, `XGBoost` 에 대한 GPU 지원도 적용하였습니다.

도커 허브(Docker Hub)에 이미지를 업로드 해 놓았습니다. 이미지를 Pull 하면 스트레스 없이 GPU가 지원되는 딥러닝 서버를 구축할 수 있습니다. 아직 CPU 버전의 이미지는 만들지 않고 있으나, 추후 요청이 있다면 만들어 볼 계획도 가지고 있습니다. 

앞으로 분기에 한 번씩은 최신 라이브러리로 업데이트 작업을 진행할 예정입니다.

자세한 사용법은 아래를 참고해 주시기 바랍니다 😁

- **GPU** 버전 도커 **Hub** 주소: [teddylee777/deepko](https://hub.docker.com/repository/docker/teddylee777/deepko)
- **GitHub** 주소: [github.com/teddylee777/deepko](https://github.com/teddylee777/deepko)



## deepko

**deepko** (**DEEP** learning docker for **KO**rean) 는 <u>파이썬(Python) 기반의 데이터 분석 / 머신러닝 / 딥러닝 도커(docker)</u> 입니다.

- 파이썬 기반의 데이터 분석, 머신러닝, 딥러닝 프레임워크의 상호 의존성 충돌을 해결 후 배포합니다.

- **한글 폰트, 한글 자연어 처리(형태소 분석기)** 를 위한 라이브러리가 사전에 설치되어 있습니다.

- **GPU** 를 지원합니다 (`LightGBM`, `XGBoost`, `PyTorch`, `TensorFlow`).

- 도커를 통한 빠른 설치와 실행이 가능합니다.

  

## 개요

TensorFlow 2.11.0 의 [tensorflow/tensorflow:2.11.0-gpu-jupyter](https://hub.docker.com/layers/tensorflow/tensorflow/2.11.0-gpu-jupyter/images/sha256-fc519621eb9a54591721e9019f1606688c9abb329b16b00cc7107c23f14a6f24?context=explore)의 도커를 베이스로 확장하여 GPU 전용 Docker파일(`gpu.Dockerfile`)을 구성하였습니다. 

TensorFlow에서 유지보수하고 있는 `2.11.0-gpu-jupyter` 도커의 경우 한글 형태소 분석기나 한글폰트, 그 밖에 PyTorch를 비롯한 여러 머신러닝/딥러닝 라이브러리가 제외되어 있기 때문에 필요한 라이브러리를 추가 설치하고 의존성에 문제가 없는지 확인한 후 배포하는 작업을 진행하고 있습니다.

본 Repository를 만들게 된 계기는 안정적으로 업데이트 되고 있는 `tensorflow/tensorflow-gpu-jupyter`에 기반하여 한글 폰트, 한글 자연어처리 패키지(konlpy), 형태소 분석기(mecab), Timezone 등의 설정을 추가하여 별도의 한글 관련 패키지와 설정을 해줘야 하는 번거로움을 줄이기 위함입니다.

- **GPU** 버전 도커 **Hub** 주소: [teddylee777/deepko](https://hub.docker.com/repository/docker/teddylee777/deepko)
- **GitHub** 주소: [github.com/teddylee777/deepko](https://github.com/teddylee777/deepko)



## 테스트된 도커 환경

- OS: Ubuntu 18.04
- GPU: RTX3090 x 2 way
- **CUDA: 11.2~11.4**
- Python (anaconda): 3.8



## 한글 관련 추가 패키지

- apt 패키지 인스톨러 카카오 mirror 서버 추가
- Nanum(나눔) 폰트, D2Coding 폰트 설치
- matplotlib 에 나눔폰트, D2Coding 폰트 추가
- mecab 형태소 분석기 설치 및 파이썬 패키지 설치
- [konlpy](https://konlpy-ko.readthedocs.io/ko/v0.4.3/): 한국어 정보처리를 위한 파이썬 패키지
- `jupyter_notebook_config.py` : Jupyter Notebook 설정 파일 추가



## 설치된 주요 라이브러리

```
catboost                     1.1.1
fastai                       2.7.10
fasttext                     0.9.2
folium                       0.14.0
gensim                       4.3.0
graphviz                     0.20.1
huggingface-hub              0.11.1
hyperopt                     0.2.7
jupyter                      1.0.0
jupyterlab                   3.5.2
kaggle                       1.5.12
keras                        2.11.0
konlpy                       0.6.0
librosa                      0.9.2
lightgbm                     3.3.4
matplotlib                   3.6.3
mecab-python                 0.996-ko-0.9.2
mlxtend                      0.21.0
nltk                         3.8.1
numpy                        1.23.5
opencv-python                4.7.0.68
optuna                       3.0.5
pandas                       1.5.2
Pillow                       9.4.0
plotly                       5.12.0
prophet                      1.1.1
PyMySQL                      1.0.2
scikit-image                 0.19.3
scikit-learn                 1.2.0
scipy                        1.8.1
seaborn                      0.12.2
sentencepiece                0.1.86
spacy                        3.4.4
SQLAlchemy                   1.4.46
tensorboard                  2.11.0
tensorflow                   2.11.0
tensorflow-datasets          4.8.1
tokenizers                   0.13.2
torch                        1.10.1+cu111
torchaudio                   0.10.1+rocm4.1
torchsummary                 1.5.1
torchtext                    0.11.1
torchvision                  0.11.2+cu111
tqdm                         4.64.1
transformers                 4.25.1
wandb                        0.13.9
wordcloud                    1.8.2.2
xgboost                      2.0.0.dev0
```



## GPU 지원 라이브러리

다음의 라이브러리에 대하여 **GPU를 지원**합니다.

1. `LightGBM` (3.3.4)
2. `XGBoost` (2.0.0.dev0)
3. `PyTorch` (1.10.1) + CUDA 11.1
4. `TensorFlow` (2.11.0) + CUDA 11.2



## 실행 방법

### STEP 1: Docker가 사전에 설치되어 있어야 합니다.

도커의 설치 및 사용법에 대하여 궁금하신 분들은 [Docker를 활용하여 딥러닝/머신러닝 환경 구성하기](https://teddylee777.github.io/linux/docker%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%98%EA%B2%BD%EA%B5%AC%EC%84%B1.md) 글을 참고해 주세요.

```bash
# step 1: apt-get 업데이트
sudo apt-get update

# step 2: 이전 버젼 제거
sudo apt-get remove docker docker-engine docker.io

# step 3: 도커(Docker) 설치 
sudo apt install docker.io

# step 4: 도커 서비스 시작
sudo systemctl start docker
sudo systemctl enable docker

# step 5: 잘 설치 되어있는지 버젼 체크
docker --version
```



### STEP 2: 도커 이미지 pull 하여 서버 실행

상황에 따라 다음 4가지 중 하나의 명령어를 실행하여 도커를 실행합니다. 세부 옵션은 아래를 참고해 주세요.

- `--rm`: 도커가 종료될 때 컨테이너 삭제

- `-it`: 인터랙티브 TTY 모드 (디폴트로 설정)

- `-d`: 도커를 백그라운드로 실행

- `-p`: 포트 설정. **local 포트:도커 포트**

- `-v`: local 볼륨 마운트. **local 볼륨:도커 볼륨**

- `--name`: 도커의 별칭(name) 설정

  

1. `Jupyter Notebook` 을 **8888번 포트로 실행**하려는 경우

```bash
docker run --runtime nvidia --rm -it -p 8888:8888 teddylee777/deepko:latest
```



2. `jupyter notebook` 서버 실행과 동시에 **로컬 volume 마운트**

```bash
docker run --runtime nvidia --rm -it -p 8888:8888 -v /data/jupyter_data:/home/jupyter teddylee777/deepko:latest
```



3. 도커를 **background에서 실행**하는 경우 (터미널을 종료해도 서버 유지)

```bash
docker run --runtime nvidia --rm -itd -p 8888:8888 teddylee777/deepko:latest
```



4. 도커를 실행 후 **bash shell로 진입**하려는 경우

```bash
docker run --runtime nvidia --rm -it -p 8888:8888 teddylee777/deepko:latest /bin/bash
```



**[참고]**

`jupyter_notebook_config.py` 을 기본 값으로 사용하셔도 좋지만, 보안을 위해서 **비밀번호 설정**은 해주시는 것이 좋습니다.

**비밀번호 설정** 방법, **방화벽 설정** 방법은 [Docker를 활용하여 딥러닝/머신러닝 환경 구성하기](https://teddylee777.github.io/linux/docker%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%98%EA%B2%BD%EA%B5%AC%EC%84%B1.md) 글의 STEP 5, 7을 참고해 주세요.



## [선택] .bashrc에 단축 커멘드 지정

`~/.bashrc`의 파일에 아래 커멘드를 추가하여 단축키로 Docker 실행


```bash
kjupyter{
    docker run --runtime nvidia --rm -itd -p 8888:8888 -v /data/jupyter_data:/home/jupyter --name dl-ko teddylee777/deepko
}
```



 위와 같이 `~/.bashrc` 파일을 수정 후 저장한 뒤 `source ~/.bashrc`로 파일의 내요을 업데이트 합니다.

추후, 긴 줄의 명령어를 입력할 필요 없이 단순하게 아래의 명령어로 도커를 백그라운드에서 실행 할 수 있습니다.

```bash
kjupyter
```




