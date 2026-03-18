# pip install ragas langchain openai

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)

# 1. 환경 변수 설정 (OpenAI 모델을 평가자로 사용 시)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 2. 평가용 데이터셋 준비
# RAG 파이프라인을 실행해서 얻은 결과물들을 아래 형식으로 모읍니다.
data_samples = {
    'question': ['대한민국의 수도는 어디인가요?'],
    'answer': ['대한민국의 수도는 서울입니다.'],
    'contexts' : [['서울은 대한민국의 최대 도시이자 수도입니다.']],
    'ground_truth': ['서울']
}

dataset = Dataset.from_dict(data_samples)

# 3. 측정 지표 선택 및 평가 실행
# faithfulness: 답변이 문맥에 충실한가?
# answer_relevance: 답변이 질문과 관련 있는가?
# context_precision: 검색된 문맥이 질문에 적합한가?
# context_recall: 정답을 맞히기에 충분한 정보가 문맥에 있는가?

score = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
    ],
)

# 4. 결과 출력
df = score.to_pandas()
print(df)