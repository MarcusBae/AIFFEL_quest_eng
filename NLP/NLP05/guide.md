# 7. 프로젝트: KoChatGPT 업그레이드 하기

### 프로젝트 개요

* 기존 KoChatGPT에 개선 전략 적용해 보기

### 주요 개선 포인트 및 전략

* 데이터셋 정제
    * 기존 KoChatGPT 모델에 사용한 데이터셋은 아직 완벽히 정제되지 않음. 
* 다양한 benchmark 데이터셋 적용
    * Human Feedback이 반영된 데이터셋을 대체하기 위해 SFT와 RM 모델에 사용할 다양한 benchmark 데이터셋 적용. 
* 최선의 디코딩을 위한 하이퍼파라미터 서치
    * 언어모델의 생성능력을 좌우하는 최선의 디코딩을 위한 하이퍼파라미터 서치가 필요함.
* 정량적인 메트릭 도입
    * 생성된 답변에 대한 주관적인 평가를 보완할 수 있는 정량적인 메트릭 도입.
* Instruction Tuning 및 Prompting 기법 적용
    LLM Trend Note1에서 살펴본 다양한 Instruction Tuning 및 Prompting 기법들도 적용해볼만 함.
* 더 큰 파라미터 스케일을 가진 모델을 사용
    * foundation model로 사용한 KoGPT-2는 Emergent abilities를 기대하기엔 다소 작은 사이즈의 모델임, 더 큰 파라미터 스케일을 가진 모델을 사용
* LoRA의 적용 또는 새로운 Instruction Tuning 및 reward ranking 알고리즘을 도입
    * 더 효율적인 연산을 수행할 수 있는 LoRA의 적용 또는 새로운 Instruction Tuning 및 reward ranking 알고리즘을 도입

---

### 실행 예시 1: 기존 데이터셋 추가 정제

* data_kochatgpt 폴더에는 세 파일이 있습니다.
    * kochatgpt_1_SFT.jsonl : SFT를 위한 prompt와 completion 문장셋
    * kochatgpt_1_RM.jsonl : RM 학습을 위한 prompt와 세 가지 ranking 문장셋
    * kochatgpt_1_PPO.jsonl : promt 문장


* 각 말뭉치를 EDA하여 도메인과 문체, 길이분포, 문장의 완성도 등을 분석합니다.
    * 언어모델의 문장생성능력은 말뭉치의 전처리 수준에 큰 영향을 받습니다.
    * 말뭉치의 분석결과를 토대로 데이터를 정제하여 모델을 재학습시켜봅니다.
    * (정제후 데이터셋 크기가 줄어들지 않도록, 다양한 augmentation 기법을 활용하여 크기를 유지 내지 증량합니다.)
    * 추가 전처리 후, 기존 인퍼런스 결과와 성능을 비교해봅니다.
    * (주관적인 평가와 BLEU, ROUGE 등을 활용한 정량적인 평가 결과를 비교 분석하여 제시합니다.)

### 실행 예시 2: 새로운 데이터셋 추가

* KoChatGPT는 human feedback이 반영된 데이터를 직접 사용하는 대신 ChatGPT API를 사용하는 대안을 선택함. 
    * LLM Trend Note1 처럼 Anthropic의 RLHF는 StackExchange 같은 온라인 상의 댓글정보를 활용하여 ranking dataset을 구축해 구현됨.
    * 비슷한 로직을 적용해볼 수 있음.
        * 하나의 prompt에 다양한 수준의 품질로 댓글이 달린 한국어로 된 웹사이트를 찾아봅시다.
        * 웹크롤링 기법을 사용해 reward 점수를 차등적으로 적용해볼 수 있는 instruction dataset과 ranking dataset을 구축.
        * KorQuAD 2.0 같은 한국어 이해 benchmark를 활용해 고품질의 데이터셋을 확보하고, KoGPT-2를 사용해 빠르게 저품질 데이터셋을 페어링해볼 수도 있음.
        * 다양한 데이터 증량전략을 구사하여 기존 데이터셋에 새로 구축한 데이터셋을 추가해 모델을 재학습시키고 추론 결과를 비교해 분석하여 제시해보기.

### 실행 예시 3: foundation model 교체

* 현재 제공되는 LMS GPU 사양으로는 수십 billion 단위 이상의 LLM을 튜닝하기 어려움.
    * 경량화, 최적화 라이브러리를 사용해 보기. 
        * 허깅페이스에서 제공하는 큰 규모의 모델을 적은 컴퓨팅 자원으로도 사용할 수 있게 해주는 경량화, 최적화 라이브러리를 사용하면 속도는 느리지만 우리의 LMS에서도 학습 및 추론이 가능해질 수 있음.
        * (힌트 : LLM Trend Note1 노드의 마지막 스텝을 참고해보세요)

    * 허깅페이스에서 제공되는 1.2B 사이즈의 한국어 GPT pretrain model로 skt/ko-gpt-trinity-1.2B-v0.5 로 foundation model을 교체해보기.
    * (단 OOM 문제를 해소하기 위해 허깅페이스에서 제공하는 다양한 training argument들을 조합하여 최상의 하이퍼파라미터를 찾아내야 합니다.)
    * 데이터셋을 아예 바꿔 모델 선택의 폭을 늘려보는 것도 좋은 선택지입니다.
    * foundation model 교체에 성공했다면, generator 함수를 수정하여 모델 인퍼런스 결과를 제시해보세요.

---

### 참고 사항

* LLM Trend Note2에서 살펴본 chatgpt 코드는 빠른 baseline모델 설계 실습을 위해 오리지널 코드를 일부 수정한 버전임.
* 프로젝트 진행을 위해 모델을 커스터마이징할 때, 필요시 "colossalai_ChatGPT_230319" 폴더 내의 원본 스크립트들을 참고할 수 있음.

---