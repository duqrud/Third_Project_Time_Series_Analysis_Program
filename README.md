# &#128250; Deep Learning 기반 시계열 분석

[TOC]

## 소개

> 팀 이름: Five Q (5Q)
>
> 프로젝트 명: Deep Learning 기반 시계열 분석
>
> 팀장 : 김재경 (GUI, 데이터 분석(LSTM))
>
> 팀원: 구준모(GUI, 데이터 분석(LSTM)), 김경수(GUI, 데이터 분석(ARIMA)), 변찬석(GUI, exe 파일 배포), 윤종현(GUI, exe 파일 배포)

## 주제

>딥러닝 기반 주식(시계열) 예측 프로그램

## 개요

> 사용자가 원하는 주식 데이터를 전처리 및  ARIMA, LSTM 학습을 활용한 예측 프로그램

## 기능

> 1. 데이터 전처리
> 2. ARIMA, LSTM 학습 (인자값 사용자가 설정할 수 있게 하여 자유도를 높힘)
> 3. 학습시킨 모델을 불러와서 데이터 예측
> 4. 전처리된 데이터, 예측된 데이터, 학습시킨 모델 저장 기능

## 향후 전망

> 1. 다른 모든 시계열 데이터도 적용 가능
> 2. 외부 요인도 인자에 포함시켜 정확도를 높히도록 함
> 3. 전처리 부분 주성분 분석 포함 
> 4. 다양한 학습 모델 추가

## 기술 스택

> PyQt5
>
> QtDesigner
>
> Pandas
>
> Numpy
>
> Tensorflow
>
> Keras
>
> Python

## 기술 설명

> PyQt5, QtDesigner: GUI 작업
>
> Pandas, Numpy: 자료구조, 데이터 전처리
>
> Tensorflow, Keras: LSTM 딥러닝을 위한 라이브러리
>
> Python: 사용 프로그래밍 언어

### 디렉토리 구조도

> ```bash
> s03p31b204
> ├── Final
> |	├── _uiFiles
> |       ├── Main.ui
> |       ├── Data_Dialog_test.ui
> |       ├── Train_Dialog.ui
> |       └── Data_Predict1.ui
> |	├── Data
> |	├── images
> |	├── Python
> |       ├── Main.py
> |       ├── Data_Dialog.py
> |       ├── Train_Dialog.py
> |       ├── LS.py
> |       └── Data_Predict.py
> |	├── save
> ├── 산출물
> ├── 메뉴얼.md
> └── README.md
> ```
>
> #### Final
>
> - 작업 폴더
>
> #### _uiFiles
>
> - Main.ui - 메인페이지 UI
> - Data_Dialog_test.ui - 전처리 페이지 UI
> - Train_Dialog.ui - 학습 페이지 UI
> - Data_Predict1.ui - 예측 페이지 UI
>
> #### DATA
>
> - 학습할 CSV 데이터를 모아놓은 폴더
>
> #### images
>
> - GUI에 사용된 이미지를 모아놓은 폴더
>
> #### Python
>
> - Main.py - 메인 페이지.py
> - Data_Dialog.py - 전처리 페이지.py
> - Train_Dialog.py - 학습 페이지.py
> - LS.py - LSTM 함수 모음.py
> - Data_Predict.py - 예측 페이지.py
>
> #### save
>
> - 모델을 저장해놓을 폴더
>
> #### 산출물
>
> - 와이어프레임, PPT

## 실행

> 메뉴얼 참조
>

## :star: 개발규칙

>branch
>
>```
>master -> develop -> feature/작업명
>```
>
>merge
>
>```
>merge 전 코드 리뷰
>충돌 시 상의 하 merge
>```
>
>commit
>
>```
>가능한 1 day 1 commit
>진행상황 | 해당하는 JIRA Story
>```

