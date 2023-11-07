# Adaptive-PT
- BERT/ELECTRA 사전학습 task 그대로 데이터만 추가하여 추가학습

```
1. 추가 학습하고자 하는 model의 vocab을 추가 데이터에 맞게 업데이트 -> source/make_vocab.py로 vocab file 생성

2. tokenizer에 적용.
```
## Environment
- ubuntu 20.04
- python 3.9.12
- docker image
```
docker pull ayaanayaan/ayaan_nv
```

## Requirements
- pytorch 1.10
- pymongo 4.1.1

## source
- bert.py : BERT 사전학습 task 코드
- electra.py : ELECTRA 사전학습 task 코드
- early_stopping.py : loss 떨어지면 checkpoint 저장
- make_vocab.py : 새로 학습할 데이터에서 vocab만들어 기존 모델 vocab file 업데이트
- pretrain_bert.py : bert 추가학습 실행 코드
- pretrain_electra.py : electra 추가학습 실행 코드