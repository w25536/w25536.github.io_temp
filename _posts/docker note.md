---
layout: page
title: "torchtext를 활용한 텍스트 데이터 전처리 방법"
description: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
headline: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-18
---



```bash

NAME                   DESCRIPTION                                     STARS     OFFICIAL
mysql                  MySQL is a widely used, open-source relation…   15583     [OK]
bitnami/mysql          Bitnami container image for MySQL               123       
circleci/mysql         MySQL is a widely used, open-source relation…   30        
cimg/mysql                                                             3         
bitnamicharts/mysql    Bitnami Helm chart for MySQL                    0         
ubuntu/mysql           MySQL open source fast, stable, multi-thread…   66        
elestio/mysql          Mysql, verified and packaged by Elestio         1         
google/mysql           MySQL server for Google Compute Engine          25        
docksal/mysql          MySQL service images for Docksal - https://d…   0         
alpine/mysql           mysql client                                    3         
mysql/mysql-server     Optimized MySQL Server Docker images. Create…   1025      
jumpserver/mysql                                                       1         
datajoint/mysql        MySQL image pre-configured to work smoothly …   2         
ddev/mysql             ARM64 base images for ddev-dbserver-mysql-8.…   1         
mysql/mysql-router     MySQL Router provides transparent routing be…   28        
mirantis/mysql                                                         0         
ilios/mysql            Mysql configured for running Ilios              1         
corpusops/mysql        https://github.com/corpusops/docker-images/     0         
mysql/mysql-cluster    Experimental MySQL Cluster Docker images. Cr…   100       
javanile/mysql         MySQL for development                           0         
vulhub/mysql                                                           1         
mysql/mysql-operator   MySQL Operator for Kubernetes                   1         
vitess/mysql           Lightweight image to run MySQL with Vitess      1         
nasqueron/mysql                                                        1         
encoflife/mysql                                                        0    
```



docker search --limit 5 nginx


```bash
NAME                              DESCRIPTION                                     STARS     OFFICIAL
nginx                             Official build of Nginx.                        20512     [OK]
nginx/nginx-ingress               NGINX and  NGINX Plus Ingress Controllers fo…   100       
nginx/nginx-prometheus-exporter   NGINX Prometheus Exporter for NGINX and NGIN…   46        
nginx/unit                        This repository is retired, use the Docker o…   64        
nginx/nginx-ingress-operator      NGINX Ingress Operator for NGINX and NGINX P…   2  
```



docker search node.js

```bash


NAME                     DESCRIPTION                                     STARS     OFFICIAL
devakumary/node.js       DockerImg                                       0         
pomeo/node.js            Node.js                                         0         
nano/node.js                                                             11        
dockernano/node.js                                                       0         
lisuo/node.js            node.js socket.io                               0         
iovietnam/node.js                                                        0         
binsix/node.js           build image node.js 8.1.2                       0         
patrckbrs/node.js        Container Node.JS 4.5 based on Hypriot/Node.…   0         
greatbsky/node.js        docker images of node.js base on centos7        0         
jack1001/node.js                                                         0         
euhariharan/node.js      Deploy node.js application                      0         
abortu/node.js           Node.JS                                         0         
zapier/node.js           The untrusted repo that I can actually tag      0         
hightail/node.js                                                         0         
adamjimenez/node.js                                                      0         
abarbu/node.js           node.js with the Android SDK                    0         
dscho/node.js                                                            0         
aooj/node.js                                                             0         
vlbitb/node.js                                                           0         
vchinchambekar/node.js   node.js image - "hello world"                   0         
makersbrand/node.js                                                      0         
bwits/node.js                                                            0         
manojkumar/node.js                                                       0         
sqerison/node.js                                                         0         
hyalx/node.js            Nodejs                                          0   
```



docker image pull mongo:latest


```bash
latest: Pulling from library/mongo
14c5b23bf5cf: Download complete 
da089b883a54: Download complete 
8bb55f067777: Download complete 
216f2761441a: Download complete 
075a696718bb: Download complete 
6e2b33d58e9a: Download complete 
50af98a999ca: Download complete 
019eaf26198f: Download complete 
Digest: sha256:4f93a84f7d4d8b1b6cb7e0c172d8a44b0bed9b399f207165ea19473bdb5a36b0
Status: Downloaded newer image for mongo:latest
docker.io/library/mongo:latest
```



https://www.docker.com/products/docker-hub/





docker container run --rm -p 3000:3000 firstnode



docker build -t firstnode .

docker container run --rm -p 3000:3000 firstnode




![[CleanShot 2025-01-14 at 11.47.53@2x.png]]





docker container run --rm -p 3000:3000 firstnode


docker container ls -a




docker container run -p 3000:3000 firstnode 


```bash
> mytestexpressapp@0.0.0 start /app
> node ./bin/www
```






docker container stop 

```bash
CONTAINER ID   IMAGE       COMMAND                  CREATED          STATUS                     PORTS     NAMES
8f9896138d4b   firstnode   "docker-entrypoint.s…"   33 seconds ago   Exited (0) 8 seconds ago             practical_hopper
```




docker container run -it ubuntu



- `docker container ls`를 실행해 현재 동작 중인 컨테이너를 확인함.
    
    - **CONTAINER ID**: 27bfb1cfb45a
    - **IMAGE**: firstnode
    - **NAMES**: firstnode
    - **PORTS**: 80→3000 (호스트 80번 포트를 컨테이너 3000번 포트와 연결)
- `docker container stop firstnode` 명령어로 **firstnode** 컨테이너를 중지(stop).
    
- `docker container rm firstnode` 명령어로 **firstnode** 컨테이너를 삭제(remove).
    
- 다시 `docker container ls` 명령어 실행 후, 현재 동작 중인 컨테이너가 없음을 확인.



**정리**

1. `docker container run -p 80:3000 --name firstnode -d firstnode`
    
    - 이미지 **firstnode**를 이용해 컨테이너를 생성 및 실행
    - 호스트의 **80번 포트**를 컨테이너의 **3000번 포트**와 연결
    - 컨테이너 이름(**--name**)을 **firstnode**로 설정
    - 백그라운드(**-d**)로 실행
2. 실행 결과로 컨테이너 ID **cee09e88c8a5b657fdde68a80d715d6b9b37b1846d556c8fef765e1a2eb5e06a**가 출력됨.
    
3. `docker container ls`
    
    - 현재 실행 중인 컨테이너 목록에서, 컨테이너 ID가 **cee09e88c8a5**, 이미지가 **firstnode**, 이름이 **firstnode**(출력상 `firstnod`으로 보이지만 동일)임을 확인
    - **0.0.0.0:80->3000/tcp**로 포트 바인딩 확인




➜  guestbook git:(main) ✗ docker container ls --filter "name=firstnode"


CONTAINER ID   IMAGE       COMMAND                  CREATED         STATUS         PORTS                          NAMES
cee09e88c8a5   firstnode   "docker-entrypoint.s…"   8 minutes ago   Up 8 minutes   80/tcp, 0.0.0.0:80->3000/tcp   firstnode



dodocker container ls --format "{{.ID}}, {{.Names}}, {{.Image}}"




➜  Desktop docker cp ./test.js f55d69b9dbf5:./home/ubuntu/test-from-host.js
Successfully copied 1.54kB to f55d69b9dbf5:./home/ubuntu/test-from-host.js
➜  Desktop





아래는 진행하신 과정을 간단히 정리한 내용입니다:

1. **이미지 다운로드 및 컨테이너 실행**
    
    - 명령어:
        
        bash
        
        코드 복사
        
        `docker container run -d -p 3306:3306 \   -e MYSQL_ALLOW_EMPTY_PASSWORD=true \   --name mysql \   mariadb:10.9`
        
    - `mariadb:10.9` 이미지를 로컬에 없으므로 Docker Hub에서 자동 다운로드(Pull)
    - 포트 매핑: 호스트의 `3306` → 컨테이너 내부 `3306`
    - 환경 변수: `MYSQL_ALLOW_EMPTY_PASSWORD=true` (비밀번호 없이 MariaDB 접속 허용)
    - 컨테이너 이름: `mysql`
2. **MariaDB 컨테이너 실행 후 로그**
    
    - 다운로드된 이미지의 Digest 정보, Pull된 레이어(이미지 구성 요소) 목록이 표시됨
    - 컨테이너가 정상적으로 실행되면서 ID가 출력됨(`30bba9bbf868...` 등)
3. **컨테이너 내부 MariaDB 접속**
    
    - 명령어:
        
        bash
        
        코드 복사
        
        `docker container exec -it mysql mysql`
        
    - 실행 중인 `mysql` 컨테이너 안으로 들어가 MariaDB 클라이언트(`mysql`) 실행
    - MariaDB Monitor 접속 확인, 버전 정보(`Server version: 10.9.8-MariaDB`) 출력
4. **정상 동작 확인**
    
    - MariaDB Shell에서 `help;`, `\h`로 명령어 도움말 확인 가능
    - 비밀번호 없이 접속되었으므로 실제 운영 환경에서는 **보안 설정**(루트 비밀번호 지정 등)을 반드시 고려 필요

이로써 **MariaDB 10.9 컨테이너**를 실행하고, **클라이언트로 접속**해 DB에 접근할 수 있음을 확인하였습니다.



```bash
MariaDB [(none)]> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
4 rows in set (0.001 sec)

MariaDB [(none)]> 
```




https://bit.ly/4h409gP




```bash
docker container run -d -p 3306:3306 \
-e MYSQL_ALLOW_EMPTY_PASSWORD=true \
--name mysql \
mariadb:10.9
```



```bash
docker container run -d -p 3306:3306 \
-e MYSQL_ALLOW_EMPTY_PASSWORD=true \
--name mysql \
-v /Users/wizard/mysql:/var/lib/mysql \
mariadb:10.9
```


```bash
docker container run -d -p 3306:3306 \
  -e MYSQL_ALLOW_EMPTY_PASSWORD=true \
  --name mysql \
  -v /Users/jeongho/mysql:/var/lib/mysql \
  mariadb:10.9
```


```bash
docker exec -it mysql mysql
create database wp CHARACTER SET utf8;
grant all privileges on wp.* to wp@'%' identified by 'wp';
flush privileges;
quit
```



```bash
docker run -d -p 8080:80 \
-e WORDPRESS_DB_HOST=host.docker.internal \
-e WORDPRESS_DB_NAME=wp \
-e WORDPRESS_DB_USER=wp \
-e WORDPRESS_DB_PASSWORD=wp \
wordpress
```


docker run -d \ --name wordpress \ --network wordpress-net \ -p 8080:80 \ -e WORDPRESS_DB_HOST=my-db-host \ -e WORDPRESS_DB_USER=my-user \ -e WORDPRESS_DB_PASSWORD=my-password \ wordpress:latest







```bash
docker container rm -f $(docker container ls -qa)
```


```
docker-compose up -d
```

```
docker-compose down
```







docker build -t guestbook-app:latest .


docker push u25536/guestbook-app


docker container run -p 3000:3000 u25536/guestbook-app


--- 


dodocker run -d -p 3000:3000 --name gbTestContainer u25536/guestbook-app

## docker run 했을때 포트를 열어줘야 사이트에 연결이 된다. 

docker port gbTestContainer





```bash
docker push [다른 유저 ID]/guestbook-app  가져온다 


docker run -d -p 3000:3000 --name [컨테이너 이름] wizard1113/guestbook-app
```

-> 자신에 local에 docker 











```
az container create \
  --resource-group <ResourceGroup이름> \
  --name myContainerInstance \
  --image crfly15.azurecr.io/guestbook-app:v1 \
  --cpu 1 --memory 1.5 \
  --ports 8000 \
  --dns-name-label myuniquehostlabel
```





```bash


az login

az group create --name [grounp_name] --location koreacentral

az acr create --resource-group [grounp_name] --name [acr_name] --sku Basic

az acr login --name [acr_name]

az acr list --resource-group [grounp_name] --query "[].{acrLoginServer:loginServer}" --output table

docker image build --platform=linux/amd64 . -t crfly15.azurecr.io/guestbook-app:v1

docker image build --platform=linux/amd64 . -t [acr_address]/[app_name]]:[tag] # mac 사용자는 platform 지정하는걸 잊지 말자! 

docker push [acr_address]/[app_name]]:[tag]



```


설정 >> 엑세스 키 >> 관리 사용자 체크 
![[CleanShot 2025-01-15 at 13.55.15 1.png]]


Container Instances  >>  컨테이너 인스턴스 만들기 >> 이미지 원본에서

![[CleanShot 2025-01-15 at 13.57.46.png]]

레지스트리 >>  해당 acr_설치 >> 아까 push해서 올린 이미지 선택  >> 태그 선택 만들면 끝 

![[CleanShot 2025-01-15 at 13.58.14.png]]




---


컨테이너 레지스트리 만들기 >> 액세스 키 >> 관리자 사용 선택 




![[CleanShot 2025-01-15 at 14.03.34.png]]



```
az acr list --resource-group RG15 --query "[].{acrLoginServer:loginServer}" --output table
```



```
AcrLoginServer
--------------------
crhello15.azurecr.io
```


```bash

az acr login --name crhello15
Login Succeeded

```

Dockerfile 쪽으로 가기 

![[CleanShot 2025-01-15 at 14.36.32@2x.png]]




![[CleanShot 2025-01-15 at 14.38.37@2x.png]]

```bash
docker image build --platform=linux/amd64 . -t crhello15.azurecr.io/guestbook-app:v1
```




![[CleanShot 2025-01-15 at 14.40.49@2x 1.png]]
```bash
docker push crhello15.azurecr.io/guestbook-app:v1
```

컨테이너 인스탄스 만들기 

![[CleanShot 2025-01-15 at 14.42.42.png]]

네트워크 주소 설정법

![[CleanShot 2025-01-15 at 14.44.37.png]]