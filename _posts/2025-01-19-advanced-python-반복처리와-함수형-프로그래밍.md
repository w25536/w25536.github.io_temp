---
layout: page
title: "Advanced Python: 반복처리와 함수형 프로그래밍"
description: ""
headline: ""
tags: 
categories: python
comments: true
published: true
---
## 1. 반복 처리의 기초 개념

### Iterable과 Iterator 예시 (Iterable 객체: 반복가능한 객체, Iterator 객체: 값을 순차적으로 반환하는 객체)

Iterator
정의: Iterator는 __iter__()와 __next__() 메서드를 구현한 객체입니다. 반복 가능한 값을 순차적으로 반환할 수 있습니다.

동작 방식:

iter() 함수로 반복 가능한 객체를 Iterator로 변환합니다.
next()를 호출할 때마다 다음 값을 반환하며, 더 이상 반환할 값이 없으면 StopIteration 예외를 발생시킵니다.

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

    1
    2
    3
    4

### Generator 예시

정의: Generator는 함수 안에 yield 키워드를 사용해 값을 반환하는 함수로, 자동으로 Iterator를 생성합니다.

동작 방식:

yield는 함수 실행을 중단하고 값을 반환하며, 이후 호출 시 중단된 지점부터 실행을 재개합니다.
next()를 통해 값을 하나씩 반환하며, 마지막 값을 반환한 후에는 StopIteration 예외를 발생시킵니다.

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

    5
    4
    3
    2
    1
    Liftoff!

## 2. 메모리 효율적 반복 처리

### Lazy Evaluation과 Generator 표현식 예시

Lazy Evaluation은 필요한 시점에 값을 계산하는 전략입니다.
데이터를 미리 생성하거나 계산하지 않고, 값이 실제로 요구될 때 계산을 수행하여 리소스를 효율적으로 사용합니다.
이 방식은 메모리와 처리 시간을 절약하는 데 유용합니다.

```python
import sys

# 일반 리스트 컴프리헨션과 제너레이터 표현식의 메모리 사용량을 비교합니다.
list_comprehension = [x * x for x in range(10000)]  # 리스트 컴프리헨션을 사용하여 0부터 9999까지의 제곱을 리스트로 생성
generator_expression = (x * x for x in range(10000))  # 제너레이터 표현식을 사용하여 0부터 9999까지의 제곱을 생성

# 각 객체의 메모리 크기를 출력합니다.
print(f"List comprehension memory size: {sys.getsizeof(list_comprehension)} bytes")  # 리스트 컴프리헨션의 메모리 크기 출력
print(f"Generator expression memory size: {sys.getsizeof(generator_expression)} bytes")  # 제너레이터 표현식의 메모리 크기 출력

# 제너레이터 표현식을 사용하여 값을 순차적으로 생성하고 출력합니다.
for value in generator_expression:
    if value > 100:  # 값이 100을 초과하면 반복을 중단
        break
    print(value)  # 현재 값을 출력
```

    List comprehension memory size: 85176 bytes
    Generator expression memory size: 208 bytes
    0
    1
    4
    9
    16
    25
    36
    49
    64
    81
    100

## 3.데이터 변환 및 필터링

### Comprehensions 예시

Comprehensions는 Python에서 반복(iteration)을 사용해 간결하게 새로운 데이터를 생성하는 문법입니다. 일반적으로 리스트, 딕셔너리, 세트 등을 만들 때 사용되며, 가독성이 좋고 간결한 코드 작성을 도와줍니다.

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

    Even squares: [4, 16, 36, 64, 100, 144, 196, 256, 324]
    Word lengths: {'apple': 5, 'banana': 6, 'cherry': 6, 'date': 4}
    Unique squares: {1, 4, 9, 16, 25}

### map(), filter(), reduce() 함수 예시

- map()은 함수를 반복 가능한 객체(리스트, 튜플 등)의 모든 요소에 적용하고, 그 결과를 새로운 map 객체로 반환합니다.
- filter()는 조건을 만족하는 요소만 추출해 새로운 filter 객체로 반환합니다.
- reduce()는 주어진 함수로 반복 가능한 객체의 모든 요소를 누적 연산해 단일 값으로 반환합니다.

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

    Doubled numbers: [2, 4, 6, 8, 10]
    Even numbers: [2, 4]
    Sum of numbers: 15

## 4. 함수형 프로그래밍 기법

### 람다 함수와 고차 함수 예시

- 람다 함수는 Python에서 lambda 키워드를 사용해 작성하는 익명 함수입니다. 이름 없이 일시적으로 사용되며, 간단한 함수를 정의할 때 유용합니다.
- 고차 함수는 다른 함수를 인자로 받거나, 함수를 반환하는 함수를 말합니다.
함수가 일급 객체(First-Class Object)로 취급되는 언어에서 사용 가능하며, Python은 이에 해당합니다.

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

    Sorted pairs by second element: [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
    Squared numbers: [1, 4, 9, 16, 25]

## 5. 자원 관리와 코드 개선 기법

### Context Manager (with 문) 예시

Context Manager는 리소스를 열고 닫는 작업(예: 파일, 데이터베이스 연결 등)을 효율적으로 관리하고, 코드의 가독성을 높여주는 Python의 기능입니다.

Context Manager는 with 문을 사용하여 구현되며, 특정 작업을 실행하기 전후에 필요한 초기화와 정리(clean-up) 작업을 자동으로 수행합니다.

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

    Opening file example.txt
    Closing file example.txt

### Decorator (데코레이터) 예시

데코레이터는 함수를 꾸며주는 함수로, 기존 함수나 메서드의 동작을 수정하거나 확장할 때 사용됩니다. 데코레이터는 다른 함수를 감싸는 래퍼(wrapper) 함수로 동작하며, Python에서 @ 기호로 간단히 적용할 수 있습니다.

```python
import time

# 함수의 실행 시간을 측정하고 보고하는 데코레이터
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete")
        return result
    return wrapper

# 'slow_function'의 실행 시간을 모니터링하기 위해 데코레이터 적용
@timer_decorator
def slow_function():
    time.sleep(2)
    print("Function finished")

# 데코레이터가 적용된 함수 호출
slow_function()
```

    Function finished
    Function slow_function took 2.0002 seconds to complete

## 6. 반복 관련 고급 모듈

### Itertools 모듈 예시

Python의 __itertools__는 반복 가능한(iterable) 데이터 처리에 유용한 함수들을 제공하는 표준 라이브러리입니다. 주로 메모리를 효율적으로 사용하면서 반복 작업을 수행할 수 있는 도구를 제공합니다.

itertools는 다양한 조합, 순열, Cartesian 곱, 그리고 무한 반복 생성 등을 지원하며, 큰 데이터셋을 처리하거나 반복 작업을 간결하고 효율적으로 구현할 때 유용합니다.

```python
import itertools

# 1. itertools.count()를 사용한 무한 반복
print("itertools.count() 예시:")
for i in itertools.count(5, 5):
    if i > 20:
        break
    print(i)  # 출력: 5, 10, 15, 20

# 2. itertools.cycle()을 사용하여 이터러블 무한 반복
print("\nitertools.cycle() 예시:")
count = 0
for item in itertools.cycle(['A', 'B', 'C']):
    if count > 5:
        break
    print(item, end=" ")  # 출력: A B C A B C
    count += 1

# 3. itertools.chain()을 사용하여 여러 이터러블 연결
print("\n\nitertools.chain() 예시:")
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
for item in itertools.chain(list1, list2):
    print(item, end=" ")  # 출력: 1 2 3 a b c

# 4. itertools.islice()를 사용하여 이터러블 슬라이스
print("\n\nitertools.islice() 예시:")
infinite_count = itertools.count(10)
# 10부터 시작하여 5개의 숫자만 가져오기
for number in itertools.islice(infinite_count, 5):
    print(number, end=" ")  # 출력: 10 11 12 13 14

# 5. itertools.product()를 사용하여 카테시안 곱 생성
print("\n\nitertools.product() 예시:")
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
for combination in itertools.product(colors, sizes):
    print(combination)  # 출력: ('red', 'S'), ('red', 'M'), ..., ('blue', 'L')

# 6. itertools.permutations()를 사용하여 순열 생성
print("\nitertools.permutations() 예시:")
items = [1, 2, 3]
for perm in itertools.permutations(items, 2):
    print(perm)  # 출력: (1, 2), (1, 3), (2, 1), (2, 3), ...

# 7. itertools.combinations()를 사용하여 조합 생성
print("\nitertools.combinations() 예시:")
for comb in itertools.combinations(items, 2):
    print(comb)  # 출력: (1, 2), (1, 3), (2, 3)

# 8. itertools.groupby()를 사용하여 그룹화
print("\nitertools.groupby() 예시:")
data = ['A', 'A', 'B', 'B', 'C', 'C', 'C']
# groupby는 반드시 정렬된 데이터에서만 유의미함
for key, group in itertools.groupby(data):
    print(f"{key}: {list(group)}")  # 출력: A: ['A', 'A'], B: ['B', 'B'], C: ['C', 'C', 'C']
```

    itertools.count() 예시:
    5
    10
    15
    20
    
    itertools.cycle() 예시:
    A B C A B C 
    
    itertools.chain() 예시:
    1 2 3 a b c 
    
    itertools.islice() 예시:
    10 11 12 13 14 
    
    itertools.product() 예시:
    ('red', 'S')
    ('red', 'M')
    ('red', 'L')
    ('blue', 'S')
    ('blue', 'M')
    ('blue', 'L')
    
    itertools.permutations() 예시:
    (1, 2)
    (1, 3)
    (2, 1)
    (2, 3)
    (3, 1)
    (3, 2)
    
    itertools.combinations() 예시:
    (1, 2)
    (1, 3)
    (2, 3)
    
    itertools.groupby() 예시:
    A: ['A', 'A']
    B: ['B', 'B']
    C: ['C', 'C', 'C']
