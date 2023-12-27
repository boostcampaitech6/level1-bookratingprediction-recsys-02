![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Level1-BookRatingPrediction&desc=RecSys-02&fontSize=50&fontColor=FFFFFF&fontAlignY=40)
- [랩업 레포트](https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-02/blob/main/docs/RecSys_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(02%EC%A1%B0).pdf)
- [최종 발표 자료](https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-02/blob/main/docs/Recsys02-level1-BookRatingPrediction.pdf)
# 프로젝트 개요
**소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회** <br>
일반적으로 책 한 권은 원고지 기준 800~1000매 정도 되는 분량을 가지고 있습니다.
뉴스기사나 짧은 러닝 타임의 동영상처럼 간결하게 콘텐츠를 즐길 수 있는 ‘숏폼 콘텐츠’는 소비자들이 부담 없이 쉽게 선택할 수 있지만, 책 한권을 모두 읽기 위해서는 보다 긴 물리적인 시간이 필요합니다. 또한 소비자 입장에서는 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정해야 하기 때문에 상대적으로 선택에 더욱 신중을 가하게 됩니다.

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 3가지의 데이터 셋(users.csv, books.csv, train_ratings.csv)을 활용하여 이번 대회에서는 각 사용자가 주어진 책에 대해 얼마나 평점을 부여할지에 대해 예측하게 됩니다.
# 팀소개

네이버 부스트캠프 AI Tech 6기 Level 1 Recsys 2조 **투머치인포** 입니다.

<aside>
    
💡 **투머치인포TooMuchInfo의 의미**

추천 시스템 2조(Two) + 많은 데이터에서 사용자에게 필요한 정보를 추천하겠다는 포부
( CLOVA X 의 도움을 받아 작성 😊 )

</aside>

## 👋 투머치인포의 멤버를 소개합니다 👋

### 🦹‍팀원소개
|김창영|신상우|이주연|이진민|조성홍|
|:---:|:---:|:---:|:---:|:---:|
|<img src='https://github.com/TooMuch-Info/.github/assets/97018869/b49f0e25-f4ec-4a80-9334-ea148eb3da0b' height=80 width=80px></img>|<img src='https://github.com/TooMuch-Info/.github/assets/97018869/b3199d59-f98a-4e49-9231-793f74498953' height=80 width=80px></img>|<img src='https://github.com/TooMuch-Info/.github/assets/97018869/2edbf1da-2a93-472a-b92f-5040219f8d71' height=80 width=80px></img>|<img src='https://github.com/TooMuch-Info/.github/assets/97018869/5245e126-87d4-490d-af9a-cc60e25f60a0' height=80 width=80px></img>|<img src='https://github.com/TooMuch-Info/.github/assets/97018869/d30ad32a-af51-4c25-9729-f579760082d7' height=80 width=80px></img>|
|[Github](https://github.com/ChangZero)|[Github](https://github.com/sangwoonoel)|[Github](https://github.com/twndus)|[Github](https://github.com/jinmin111)|[Github](https://github.com/GangBean)|

### 👨‍👧‍👦 Team 협업
[팀 노션](https://petite-giant-ce3.notion.site/bc7381aa3bf9444dbccd95f14fa497ea?pvs=4)

[팀 깃 조직](https://github.com/TooMuch-Info)

[WandB](https://wandb.ai/ai-teah-recsys02)

### 📝 Ground Rule
개발자스럽게 깃헙 사용하기
1. 커밋 메세지 컨벤션 유다시티 스타일
2. 1이슈 1브랜치 1머지
3. 매일 진행 목표 정해서 관리
4. 데일리 스크럼/피어세션때 PR코드 리뷰 후 병합


모델 학습 및 제출 모델
- 개인별 관심 모델 구현 후 공유
- 개인별 SOTA 모델 기준 가능한 조합 앙상블

제출 방식
- 인당 2회 + 11:20PM 이후 남는 경우 선착순

<br>

# 프로젝트 개발 환경 및 기술 스택
## ⚙️ 개발 환경
- OS: Linux-5.4.0-99-generic-x86_64-with-glibc2.31
- GPU: Tesla V100-SXM2-32GB * 5
- CPU cores: 8

## 🔧 기술 스택
![](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white)
![](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=black)
![](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white)



# 프로젝트 구조
```
├── data                 # 데이터...
├── ensemble.py          # ...
├── evaluation.py
├── log
├── main.py
├── notebooks
├── requirement.txt
├── saved_models
├── src
│   ├── __init__.py
│   ├── data
│   ├── ensembles
│   ├── ml_config
│   ├── models
│   ├── train
│   └── utils.py
├── submit
├── sweep
└── wandb
```






![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=footer&)