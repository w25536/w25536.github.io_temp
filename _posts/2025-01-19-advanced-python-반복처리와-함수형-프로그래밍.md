---
layout: page
title: "Advanced Python: 반복처리와 함수형 프로그래밍"
description: ""
headline: ""
tags:
  - python
  - 파이썬
  - 파이토치
  - 전처리
  - data
  - science
  - 데이터
  - 분석
  - 딥러닝
  - 딥러닝
  - 자격증
  - 머신러닝
  - 빅데이터
categories: 
comments: true
published: true
---



```python
class CustomIterable:
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __iter__(self):
    return CustomIterator(self.start, self.end)
class CustomIterator:
  def __init__(self, start, end):
    self.current = start
    self.end = end
  def __iter__(self):
    return self
  def __next__(self):
    if self.current >= self.end:
      raise StopIteration
    else:
      self.current += 1
      return self.current - 1
# CustomIterable을 사용하여 순회하기
custom_iterable = CustomIterable(1, 5)
for value in custom_iterable:
  print(value)
```



```python
def countdown(start):
  while start > 0:
    yield start
    start -= 1
  yield "Liftoff!"
# 제너레이터 사용하기
for value in countdown(5):
  print(value)
```



```python
import sys
# 일반 리스트와 제너레이터 표현식의 메모리 사용 비교
list_comprehension = [x * x for x in range(10000)]
generator_expression = (x * x for x in range(10000))
print(f"List comprehension memory size: {sys.getsizeof(list_comprehension)} bytes")
print(f"Generator expression memory size: {sys.getsizeof(generator_expression)} bytes")
# 제너레이터 표현식의 사용 예
for value in generator_expression:
  if value > 100:
    break
  print(value)
```