---
layout: page
title: "안드로이드(Android) - AsyncTask"
description: "Android 개발을 하다보면 자주 보이지만, 아직 익숙하지는 않은 내용들을 다룰 다소 생소하고 어렵다고 생각하고 깊게 파보지 못한 내용을 선정해 한 번 파볼 예정입니다. 이번 주제는 AsyncTask입니다."
headline: "Android Util class 인 AsyncTask에 대해 파보겠습니다."
tags: android
comments: true
published: true
categories: android
---

우연치 않은 계기로 안드로이드에서 Open Source로 올려놓은 [Bitmap Displaying](https://github.com/googlesamples/android-DisplayingBitmaps)을 그대로 따라 구현해 보려고, 소스코드를 import 시킨 후, 분석을 시작했다.

**근데, 확실히 달랐다.** 

getter/setter를 남발했던 내 [boilerplate code](https://ko.wikipedia.org/wiki/%EC%83%81%EC%9A%A9%EA%B5%AC_%EC%BD%94%EB%93%9C)가 부끄러워 지기 시작했고, 소스코드를 보고 따라 해보고 스스로 구현해 보면서, 디자인 패턴 혹은 상속관계를 왜 그렇게 설정했는지 점차 알아갈 수 있었다. 

1주일 정도 분석하고 따라서 구현해 봤는데, 내가 스스로 코드를 짤 때 보다 **다른 사람의 잘 정돈된 코드를 보고 그대로 구현해 볼 때** 많은 것들을 배울 수 있었다. 

무엇보다, 생소한 Util 클래스들을 화려하게 그렇지만, ~~단순 과시용은 절대 아니고~~, 적재적소에 사용하는 것을 보면서, 평소 다소 생소했던 클래스지만, 알고나면 엄청 유용하게 활용할 수 있는 Class들을 파봐야 겠다는 생각이 들었다.

어차피, ***Android 공식 문서의 번역 수준 + 조금 더 알아보는 정도*** 겠지만, 스스로 정리를 해보면 좀 더 친숙하게 활용해 볼 수 있을 것 같은 느낌적인 느낌이랄까..

아무튼,
그 첫번째는 **AsyncTask** 이다. 

## AsyncTask

AsyncTask 는 UI thread와 Handler의 처리를 위한 Helper Class 이다.
AsyncTask는 짧은 기간의 동작 (Max. 몇 초정도) 에 적합하고, 긴 백그라운드 작업시에는 java.util.concurrent 패키지 안에 있는 *Executor, ThreadPoolExecutor* 그리고 *FutureTask*를 활용하는 것을 추천한다.

Asynchronous Task 줄여서 AsyncTask는 주로 background에서 계산된 결과값을 UI thread에서 보여줘야할 때 사용하기 위한 클래스다. 

## 상황

인터넷에서 영화 파일을 다운로드 받아야 한다고 가정해보자.

일단, 네트워크에서 파일을 다운로드 받는 과정을 UI thread에서 진행하게 된다면, 다운로드 받는 동안 사용자는 현재 다운로드 작업이 UI thread에서 진행되고 있기 때문에 화면이 멈춰버린 상황이 될 것이고,

손가락만 빨다가 UI 스레드가 약 5초 이상 차단되어 있으면 사용자에게 악명 높은...

     "애플리케이션이 응답하지 않습니다"(ANR) 
*ANR = Application Not Responding*

대화상자가 표시된다.

**(어차피 네트워크 Task는 Background Thread에서 작업해야 하고, 그렇지 않으면, 경고나 compile error가 발생한다.)**

다시 돌아와서, 사용자들에게 파일이 다운로드 되는 동안 프로그레스 바를 띄워 현재 진행상황을 알려주고, 실제 다운로드 Task는 Background에서 동작 시키고자 할 때 바로 **AsyncTask**를 활용하면 되겠다.

## 함수와 인자

인자는 3 제네릭 타입 으로 정의되는데, **Params**, **Progress**, 그리고 **Result** 이며,

단계는 4단계로 정의된다. 

간단히 정리 해보면,

**onPreExecute**: UI thread 에서 실행되고, Progress Bar를 띄울때

**doInBackground** Background thread이고 onPreExecute() 호출된 다음

**onProgressUpdate** UI thread에서 실행되고, publishProgress(Progress...)가 호출되면 실행. 주로 Progress Bar 갱신할 때 사용.

**onPostExecute** UI thread에서 실행. 모든 백그라운 작업이 종료 후 호출됨.


## 예제 코드

```java
private class DownloadFilesTask extends AsyncTask<URL, Integer, Long> {
    protected Long doInBackground(URL... urls) {
        int count = urls.length;
        long totalSize = 0;
        for (int i = 0; i < count; i++) {
            totalSize += Downloader.downloadFile(urls[i]);
            publishProgress((int) ((i / (float) count) * 100));
            // Escape early if cancel() is called
            if (isCancelled()) break;
        }
        return totalSize;
    }

    protected void onProgressUpdate(Integer... progress) {
        setProgressPercent(progress[0]);
    }

    protected void onPostExecute(Long result) {
        showDialog("Downloaded " + result + " bytes");
    }
}
```

```java
new DownloadFilesTask().execute(url1, url2, url3);
```


## 가변인자 (varargs)
여기서 주목해야할 함수 인자 중 **가변인자(varargs)**인 

doInBackground의 인자인 
    URL... urls

onProgressUpdate의 인자인
    Integer... progress

부분을 살펴보자.

좀 생소할 수 있는 포인트인데,
가변인자는 0 혹은 그 이상의 인자를 받을 수 있음을 나타낸다.
예시로 설명하자면,

```java    
thisIsFunction(); // 가능
thisIsFunction("arg1"); //가능
thisIsFunction("arg1", "arg2", "arg3"); //가능
thisIsFunction(new String[]{"arg1", "arg2", "arg3"}); //가능
```

이렇게 인자를 omit 하여도 성립하고, 1개 이상의 인자를 넘겨주거나, 배열 형태로 넘겨 줄 수 있다.
하지만, 여기서 **주의해야 할 점**은,
인자를 1개만 넘겨주어도 배열의 형태로 다뤄줘야 한다는 점이다.

다시 doInBackground 함수로 돌아가서,
urls를 1개만 넘겨주었더라도,

```java
protected Long doInBackground(URL... urls) {
    long totalSize =  Downloader.downloadFile(urls[0]);
    return totalSize;
}
```

이렇게 배열로 처리해주어야만 한다.
또한, varargs를 다른 parameter와 동시에 넘겨줄 경우
**varargs는 항상 나중에 자리해야만 한다**.

```java
thisIsFunction(int n, String... strings) // 가능
thisIsFunction(String... strings, int n) // Error
```

## AsyncTask 의 취소(cancel)

AsyncTask는 **cancel(boolean)**을 통해 취소할 수 있다.

```java
downloadTask.cancel(true);
```

AsyncTask가 취소되면, onPostExecute(Object)가 호출 되지 않고, 대신 onCancelled(Object)가 호출되게 되는데, 시점은 doInBackground(Object[])가 return 될 때이다.

Task가 cancel 되었을 때 최대한 빨리 Task가 cancelled에 대한 처리를 하게 하고 싶다면, **isCancelled()를 doInBackground 함수 내에서 주기적으로 체크**하게 해야한다.

## 추가적으로 주의해야 할 점

Android Developers의 공식 문서에는 

    onPreExecute()
    onPostExecute(Result)
    doInBackground(Params...)
    onProgressUpdate(Progress...)

를 **명시적으로 호출하지 말라**고 한다.

또한, AsyncTask의 생성과 로드, execute() 함수 호출은 UI thread에서 실행되어야 하며,

task의 실행은 한 번만 호출되어야 한다고 명시하고 있다.


참고: [AsyncTask - Android Developer](https://developer.android.com/reference/android/os/AsyncTask.html)

