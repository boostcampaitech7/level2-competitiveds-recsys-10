# 전세가 예측 프로젝트

전세는 주택 임차 계약 유형 중 하나로 세를 내지 않고 일정한 비용을 지불하고 계약 기간이 끝난 후 돌려받는 독특한 제도입니다. 수도권에서는 전세 비율이 전체 전월세 계약에서 60% 이상을 차지하며 전세 시장 동향은 부동산에서 굉장히 중요한 지표로 간주됩니다. 이번 대회는 9월 30일부터 10월 24일까지 약 4주간 진행된 대회로 2019년 4월부터 2024년 4월까지의 부동산 전세계약 관련 데이터를 바탕으로 AI와 머신러닝을 활용해 전세가를 예측하는 알고리즘을 개발하는 대회입니다.

<br/>

# Team

||<img src="https://avatars.githubusercontent.com/u/20788198?v=4" width="100" height="100"/>|<img src="https://avatars.githubusercontent.com/u/81938013?v=4" width="100" height="100"/>|<img src="https://avatars.githubusercontent.com/u/112858891?v=4" width="100" height="100"/>|<img src="https://avatars.githubusercontent.com/u/103016689?v=4" width="100" height="100"/>|<img src="https://avatars.githubusercontent.com/u/176903280?v=4" width="100" height="100"/>|
|:-:|:-:|:-:|:-:|:-:|:-:|
|공통|곽정무<br/>[@jkwag](https://github.com/jkwag)|박준하<br/>[@joshua5301](https://github.com/joshua5301)|박태지<br/>[@spsp4755](https://github.com/spsp4755)|신경호<br/>[@Human3321](https://github.com/Human3321)|이효준<br/>[@Jun9096](https://github.com/Jun9096)|
|EDA 및 모델링|Confluence<br/>템플릿 구축,<br/>회의록 작성,<br/>모델 결과 분석|기본적인 <br/>프레임워크 구현,<br/>외삽 모델 개발|AutoML,<br/>클러스터링|hyperparameter<br/>최적화 세팅,<br/>이상치 탐지|Jira 세팅,<br/>클러스터링 및<br/>피쳐 엔지니어링,<br/>랩업 리포트 관리|

<br/>

# Prerequisites and Installation
- Python 3.11
- pip

```bash
pip install -r requirements.txt
```

<br/>

# Usage

### 1\. 프로젝트 최상위 폴더에 'data' 폴더를 생성하고 안에 원시 데이터를 넣습니다.

### 2\. scrip.py에서 사용할 모델을 선택하고, 학습 데이터 및 검증 데이터를 설정합니다.

```python
# 모델을 선택합니다.
manager.select_model('geo_model.py')

# 모델을 학습하고 검증 데이터의 예측값을 받아옵니다.
true, pred = manager.validate_model(...)

# 모델을 학습하고 테스트 데이터의 예측값을 받아옵니다.
pred = manager.test_model()
```

### 3\. script.py를 실행합니다.

```bash
python script.py
```
<br/>

# Models

### - Geo model

지리 기반 feature가 추가된 LightGBM 모델입니다.

### - Naive model

이전 거래들의 가중 평균을 통해 예측하는 단순 시계열 모델입니다.

### - Ensemble model

Geo model과 Naive model을 앙상블한 모델입니다.

### - Extrapolation model

외삽을 통해 과거 전세가격을 예측하는 모델입니다.