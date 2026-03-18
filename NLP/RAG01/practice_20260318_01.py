#------------------------------------------------------------------------------
# RAGAS 사용해 보기 -  퍼실 자료 
#------------------------------------------------------------------------------

# pip install -qU ragas langchain langchain-openai langchain-community chromadb tiktoken nest_asyncio pandas

# from google.colab import userdata
# os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_KEY')


# RAGAS는 평가 속도를 높이기 위해 내부적으로 asyncio.gather를 사용해 수백 개의 API를 동시에 비동기 호출합니다.
# Jupyter/Colab 환경은 이미 자체적인 비동기 이벤트 루프를 돌리고 있어서 충돌이 발생합니다.
# nest_asyncio는 이 충돌을 막아주어 Colab에서도 RAGAS가 병렬 처리를 할 수 있게 해줍니다.

import functools
import numpy as np
import os
import tiktoken
from dotenv import load_dotenv
from operator import itemgetter

from datasets import Dataset
import pandas as pd

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_recall, context_precision,
)

load_dotenv()

import nest_asyncio
nest_asyncio.apply()

load_dotenv()

def func1():
    # 데이터 준비 
    #   question : 사용자의 질문
    #   ground truth : 질문에 맞는 정답 (= 모범 답안)
    #   answer : LLM이 생성한 답변
    #   context : LLM이 답변을 생성하기 위해 참고한 정보 (=context)
    questions = ["주가수익비율(PER)이란 무엇이며 어떻게 계산하나요?"]
    ground_truths = ["주가수익비율(PER)은 기업의 주가를 주당순이익(EPS)으로 나눈 값으로, 해당 기업의 주가가 실제 수익에 비해 얼마나 높게 형성되어 있는지를 나타내는 지표입니다. 이는 투자자들이 기업의 이익 1원을 얻기 위해 지불하는 비용을 의미하며, 보통 업종 평균 PER과 비교하여 저평가 혹은 고평가 여부를 판단합니다."]
    answers = ["PER(Price Earnings Ratio)은 현재 주가를 주당순이익(EPS)으로 나눈 비율을 의미합니다. 여기서 주가는 시장에서 거래되는 가격을 말하며, 주당순이익(EPS)은 기업이 벌어들인 순이익을 발행 주식 수로 나눈 값입니다. PER이 낮을수록 해당 기업이 벌어들이는 이익에 비해 주가가 저렴하다는 '저평가' 신호로 해석될 수 있으며, 반대로 높으면 '고평가' 혹은 미래 성장성이 높다고 판단합니다."]
    contexts = [[
        "A: 주가수익비율(PER)은 현재 주가를 1주당 예상 순이익(EPS)으로 나눈 수치로, 기업의 수익성 대비 주가 수준을 평가하는 핵심 지표입니다. "
        "* EPS(Earnings Per Share): 기업이 벌어들인 당기순이익을 발행주식 총수로 나눈 값. "
        "** 업종 평균 PER: 동일 산업군 내 기업들의 PER을 평균낸 값으로, 개별 기업의 상대적 가치를 판단하는 기준이 됨. "
        "\n\n"
        "| 종목명 | 현재가 | EPS | PER | 업종평균 PER | 투자판단 |"
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
        "| 삼성전자 | 75,000 | 5,000 | 15.0x | 18.2x | 저평가 |"
        "| SK하이닉스 | 180,000 | 9,000 | 20.0x | 18.2x | 고평가(성장세) |"
        "| 현대차 | 250,000 | 35,000 | 7.1x | 10.5x | 저평가 |"
        "| 카카오 | 50,000 | 500 | 100.0x | 45.0x | 고평가 |"
    ]]

    # To dict
    data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset = dataset,
        metrics=[
            context_precision, context_recall, faithfulness, answer_relevancy,
        ],
    )

    # result = {
    #   'context_precision': 1.0000, 
    #   'context_recall': 0.5000, 
    #   'faithfulness': 0.8000, 
    #   'answer_relevancy': nan
    # }
    print(result)

    # df = result.to_pandas()
    # print(df)

def func2():
    # 판사(Judge) 역할을 할 LLM과 임베딩 세팅
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 평가용 가상 데이터셋
    data_samples = {
        "question": [   # 사용자 질문
            "프랑스의 수도는 어디이며, 가장 유명한 미술관은 무엇인가요?",
            "블랙홀의 사건의 지평선이란 무엇인가요?"
        ],
        "contexts": [  # 답변을 생성하기 위해 참고한 정보 (=context)
            [
                "파리는 프랑스의 수도이자 세계적인 문화 중심지입니다.",
                "루브르 박물관은 파리에 위치해 있으며, 모나리자와 같은 명작을 소장하고 있습니다."
            ],
            [
                "블랙홀은 중력이 너무 강해 빛조차 빠져나갈 수 없는 천체입니다.",
                "우주 공간에는 수많은 블랙홀이 존재한다고 추정됩니다." # 의도적 누락: 사건의 지평선 설명 없음
            ]
        ],
        "answer": [  # 생성된 답변
            "프랑스의 수도는 파리이며, 가장 유명한 미술관은 루브르 박물관입니다.",
            "블랙홀의 사건의 지평선은 중력이 너무 강해 빛조차 빠져나갈 수 없는 경계면을 의미합니다. 일반 상대성 이론에 의해 예측되었습니다." # RAG 관점의 환각 발생
        ],
        "ground_truth": [  # = 모범 답안
            "프랑스의 수도는 파리이고, 대표적인 미술관은 루브르 박물관입니다.",
            "사건의 지평선(Event Horizon)은 블랙홀의 탈출 속도가 빛의 속도를 넘어서는 공간의 경계입니다."
        ]
    }

    evaluation_dataset = Dataset.from_dict(data_samples)

    # 결과 평가
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False # API 에러 발생 시 중단하지 않음
    )

    pd.set_option('display.max_colwidth', None)
    df_result = result.to_pandas()
    # display(df_result[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])

    # faithfulness  answer_relevancy  context_precision  context_recall
    # 0               0.500000          0.544565                0.5             1.0
    # 1               0.666667          1.000000                1.0             0.0
    print(df_result[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])

    # 추가
    # 정답지 자동 생성할 수 있다. 
    # https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/

    # RAG에서 metric 여러가지 있다. 
    # https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

# func2()


import requests
from bs4 import BeautifulSoup

def get_wiki_content(url):
    # 1. 웹 페이지 가져오기
    response = requests.get(url)
    if response.status_code != 200:
        return "페이지를 불러올 수 없습니다."

    # 2. HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. 본문 영역 찾기 (위키백과의 본문은 보통 'mw-content-text' 클래스 안에 있습니다)
    content = soup.find('div', {'id': 'mw-content-text'})
    
    # 4. 불필요한 요소 제거 (각주, 편집 링크, 표 등)
    for tag in content.find_all(['table', 'div', 'sup']):
        tag.decompose()

    # 5. 텍스트 추출 및 정제
    
    paragraphs = content.find_all('p')
    text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    
    return text

# 실행 예시
wiki_url = "https://ko.wikipedia.org/wiki/%EA%B0%95%EC%95%84%EC%A7%80"
wiki_text = get_wiki_content(wiki_url)

# 결과 확인 및 파일 저장
with open("dog_wiki.txt", "w", encoding="utf-8") as f:
    f.write(wiki_text)

print("본문 추출 완료! 'dog_wiki.txt' 파일이 생성되었습니다.")