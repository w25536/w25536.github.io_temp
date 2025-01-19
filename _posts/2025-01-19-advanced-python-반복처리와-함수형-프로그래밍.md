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
  - 머신러닝
  - 빅데이터
categories: python
comments: true
published: true
---

파이썬의 이터레이터(iterator)
파이썬의 이터레이터는 next() 라는 특별 메소드를 호출함으로써 컨테이너 안에 있는 값들을 차례대로 순회할 수 있는 객체이다.

iterator의 원리
파이썬의 iter() 내장 함수는 객체의 __iter__ 메소드를 호출하며, __iter__ 메소드는 해당 컨테이너의 iterator를 리턴해야 한다. 그리고 __iter__로 반환된 iterator는 __iter__와 __next__를 구현해야만 한다. 역으로 iter와 next를 둘 다 구현하고 있는 객체를 iterator의 객체라고 볼 수 있다!

iterator의 특성
iterator 객체는 반드시 __next__ 메소드를 구현해야 한다. 파이썬 내장함수 iter()는 iterable 객체에 구현된 iter를 호출함으로써 그 iterable 객체의 iterator 객체를 리턴한다. 이때 iter() 메소드가 리턴하는 iterator는 동일한 클래스 객체가 될 수도있고 따로 작성된 iterator 객체가 될 수도 있다!!



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


```python
# 리스트 컴프리헨션을 사용하여 짝수만 필터링하고 제곱 계산
numbers = range(1, 20)
even_squares = [x ** 2 for x in numbers if x % 2 == 0]
print(f"Even squares: {even_squares}")
# 딕셔너리 컴프리헨션으로 문자열 길이를 키로 하는 딕셔너리 생성
words = ["apple", "banana", "cherry", "date"]
word_lengths = {word: len(word) for word in words}
print(f"Word lengths: {word_lengths}")
# 세트 컴프리헨션으로 중복 제거
duplicates = [1, 2, 2, 3, 4, 4, 5]
unique_squares = {x ** 2 for x in duplicates}
print(f"Unique squares: {unique_squares}")
```

```python
from functools import reduce
# map()을 사용하여 모든 숫자에 2를 곱하기
numbers = [1, 2, 3, 4, 5]
doubled_numbers = list(map(lambda x: x * 2, numbers))
print(f"Doubled numbers: {doubled_numbers}")
# filter()를 사용하여 짝수만 걸러내기
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_numbers}")
# reduce()를 사용하여 모든 숫자의 합 계산하기
sum_numbers = reduce(lambda x, y: x + y, numbers)
print(f"Sum of numbers: {sum_numbers}")
```

```python
# 리스트를 정렬하면서 람다 함수 사용하기
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
# 숫자가 아닌 문자열을 기준으로 정렬
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(f"Sorted pairs by second element: {sorted_pairs}")
# 고차 함수 예제 - 함수를 인자로 전달하기
def apply_operation(numbers, operation):
  return [operation(num) for num in numbers]
# 숫자 리스트에 제곱을 적용
squared_numbers = apply_operation(numbers, lambda x: x ** 2)
print(f"Squared numbers: {squared_numbers}")
```

```python
class ManagedFile:
  def __init__(self, filename):
    self.filename = filename
  def __enter__(self):
    self.file = open(self.filename, 'w')
    print(f"Opening file {self.filename}")
    return self.file
  def __exit__(self, exc_type, exc_value, traceback):
    if self.file:
      self.file.close()
    print(f"Closing file {self.filename}")
# Context manager를 사용하여 파일 쓰기 작업 수행
with ManagedFile('example.txt') as f:
  f.write('Hello, world!\n')
  f.write('ManagedFile is working correctly.\n')
```

```python
import time
# 함수의 실행 시간을 측정하는 데코레이터
def timer_decorator(func):
    def wrapper(*args, **kwargs):
      start_time = time.time()
      result = func(*args, **kwargs)
      end_time = time.time()
      print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete")
      return result
    return wrapper
# 데코레이터를 사용하여 함수의 실행 시간 측정
@timer_decorator # is equivalent to “timer_decorator(slow_function)”
def slow_function():
  time.sleep(2)
  print("Function finished")
# 함수 호출
slow_function()
```

