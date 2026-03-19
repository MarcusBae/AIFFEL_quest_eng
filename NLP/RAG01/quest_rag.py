# RAG 시스템 비교 분석: Baseline vs Improved (Multi-Query)

# 원본 문서 : 

import os
import nest_asyncio
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from operator import itemgetter
from datasets import Dataset

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from ragas import evaluate
# from ragas.metrics.collections import (
from ragas.metrics import (
    faithfulness, answer_relevancy, context_recall, context_precision,
)

import pandas as pd

class RAGProcessor:
    def __init__(self, file_path, cfg=None):
        if cfg is None:
            cfg = {}
        load_dotenv()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.model_name = cfg.get("model_name", "gpt-4o")
        self.embedding_model_name = cfg.get("embedding_model_name", "text-embedding-3-small")
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.chunk_overlap = cfg.get("chunk_overlap", 100)
        
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
        
        # 문서 로드 및 분할
        loader = TextLoader(file_path)
        pages = loader.load() # TextLoader doesn't have load_and_split usually, it loads as a single document by default or multiple if specified

        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            length_function=self.tiktoken_len
        )
        self.texts = text_splitter.split_documents(pages)
        
        # 벡터스토어 빌드
        self.db = Chroma.from_documents(self.texts, self.embedding_model)
        
        # 프롬프트 템플릿
        template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
        self.prompt = ChatPromptTemplate.from_template(template)

    def tiktoken_len(self, text):
        return len(self.tokenizer.encode(text))

    def get_rag_chain(self, cfg):
        k = cfg.get("k", 3)
        search_type = cfg.get("search_type", "similarity")
        search_kwargs = cfg.get("search_kwargs", {"k": k})
        
        base_retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        if cfg.get("use_multi_query", False):
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.llm
            )
        else:
            retriever = base_retriever
            
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain, retriever

class RAGEvaluator:
    def __init__(self, llm=None, embeddings=None, embedding_model_name="text-embedding-3-small"):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = embeddings or OpenAIEmbeddings(model=embedding_model_name)

    def run_eval(self, data_list):
        """
        data_list: list of dicts with keys ['question', 'answer', 'contexts', 'ground_truth']
        """
        ds = Dataset.from_dict({
            "user_input": [x["question"] for x in data_list],
            "response": [x["answer"] for x in data_list],
            "retrieved_contexts": [x["contexts"] for x in data_list],
            "reference": [x["ground_truth"] for x in data_list]
        })
        return evaluate(
            ds, 
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=self.llm,
            embeddings=self.embeddings
        )


def print_cfg(cfg):
    print("\n" + "="*50)
    print(f" [Config: {cfg.get('title', 'N/A')}] ")
    print("="*50)
    for k, v in cfg.items():
        if k != "title":
            print(f" - {k}: {v}")
    print("-" * 50)


def run_rag_pipeline(evaluator, chain, retriever, questions, ground_truths, cfg):
    title = cfg.get("title", "RAG")
    print_cfg(cfg)
    results = []
    for q, gt in zip(questions, ground_truths):
        answer = chain.invoke(q)
        contexts = [doc.page_content for doc in retriever.invoke(q)]
        results.append({"question": q, "answer": answer, "contexts": contexts, "ground_truth": gt})
        
    print(f"--- Evaluating {title} ---")
    res = evaluator.run_eval(results)
    # Ragas EvaluationResult를 dict 형식으로 안전하게 변환 (평균 점수, 수치 데이터만 계산)
    return res.to_pandas().mean(numeric_only=True).to_dict()

def main():
    nest_asyncio.apply()
    
    if "__file__" in globals():
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        ROOT_DIR = os.getcwd()
    
    file_path = os.path.join(ROOT_DIR, "data/dog_wiki.txt")

    # # 1. 평가 질문 리스트
    # eval_questions = [
    #     "강아지의 정의는 무엇인가요?",
    #     "강아지가 성체 개가 되기까지 보통 어느 정도의 시간이 걸리나요?",
    #     "강아지의 사회화 시기는 대략 언제부터 언제까지인가요?",
    #     "강아지에게 사회화 교육이 중요한 이유는 무엇인가요?",
    #     "갓 태어난 강아지의 신체적 특징 중 감각에 대한 설명은 무엇인가요?",
    #     "강아지가 젖을 떼고 사료를 먹기 시작하는 시기는 언제인가요?",
    #     "강아지에게 치명적일 수 있는 음식 중 대표적인 것은 무엇인가요?",
    #     "강아지의 예방 접종은 보통 언제부터 시작하나요?",
    #     "강아지라는 단어의 어원은 무엇인가요?",
    #     "강아지를 입양할 때 환경 조성에서 가장 주의해야 할 점은 무엇인가요?"
    # ]

    # # 2. 실제 정답(Ground Truth) 리스트
    # # RAG 평가 시 실제 문서 내용과 생성된 답변을 비교하는 기준이 됩니다.
    # ground_truths = [
    #     "개의 새끼를 일컫는 말입니다.",
    #     "품종에 따라 차이가 있으나 보통 1년 정도가 지나면 성체 개가 됩니다.",
    #     "보통 태어난 후 3주부터 12주 사이를 사회화 시기라고 합니다.",
    #     "이 시기에 겪는 경험이 성견이 되었을 때의 성격과 행동 양식에 큰 영향을 미치기 때문입니다.",
    #     "눈을 뜨지 못해 앞을 볼 수 없으며, 귀도 막혀 있어 소리를 듣지 못합니다.",
    #     "보통 생후 4주에서 6주 사이부터 젖을 떼고 이유식을 시작합니다.",
    #     "초콜릿, 양파, 포도 등이 대표적인 위험 음식입니다.",
    #     "모체로부터 받은 면역력이 떨어지는 생후 6~8주 경부터 시작하는 것이 일반적입니다.",
    #     "'개'에 어린 것을 뜻하는 접미사 '-아지'가 붙어 만들어진 단어입니다.",
    #     "강아지가 삼킬 수 있는 작은 물건을 치우고, 위험한 전선 등을 정리하여 안전한 환경을 만드는 것입니다."
    # ]

    # RAG 시스템 평가를 위한 고난도 데이터셋 (위키백과 '강아지' 기반)

    eval_questions = [
        "강아지의 유치 개수와 성견의 영구치 개수 차이를 구하고, 어금니가 돋아나는 시점을 기준으로 한 성장 단계를 설명하시오.",
        "개의 기원과 관련하여 회색늑대와의 유전적 관계 및 가축화 시기에 대한 학계의 통설을 기술하시오.",
        "강아지의 감각 기능 중 청각과 후각의 특징을 인간과 비교하여 기술하고, 특히 가청 주파수의 범위를 수치로 제시하시오.",
        "문서에 언급된 '강아지'라는 단어의 어원적 구성 요소를 분석하고, 현대 국어에서 '망아지'와의 공통적인 형성 원리를 설명하시오.",
        "강아지의 지능 발달 수준을 인간의 연령과 비교하고, 지능 수준에 영향을 미치는 요인 3가지를 본문 내용에 근거해 제시하시오.",
        "강아지의 성장 과정 중 '사회화 시기'가 갖는 중요성과 이 시기에 적절한 자극을 받지 못했을 때 발생할 수 있는 부작용을 서술하시오.",
        "강아지의 신체 부위 중 발바닥(Pad)의 해부학적 기능과 땀샘의 위치가 체온 조절에 미치는 영향을 설명하시오.",
        "개의 수명에 영향을 미치는 체구(크기)별 상관관계를 설명하고, 소형견과 대형견 중 어느 쪽이 일반적으로 더 장수하는지 기술하시오.",
        "강아지의 영양 섭취에 있어 반드시 피해야 할 음식물 중 '초콜릿'과 '양파'가 신체에 미치는 치명적인 영향의 차이를 기술하시오.",
        "본문에서 '강아지'라는 명칭이 현대 사회에서 성견까지 포함한 반려견 전체를 지칭하는 용어로 확장된 국어학적 배경을 설명하시오."
    ]

    ground_truths = [
        "강아지의 유치는 28개이며, 성견의 영구치는 42개로 총 14개의 차이가 난다. 생후 4개월 무렵부터 유치가 빠지기 시작하며 영구치인 어금니가 돋아나기 시작한다.",
        "개는 약 1만 5천 년 전 또는 그 이전부터 회색늑대를 조상으로 하여 가축화되었다. 이는 인류가 농경 생활을 시작하기 전인 수렵 채집 단계에서 이루어진 것으로 보고 있다.",
        "후각 세포는 약 2억 개로 인간보다 40배 이상 발달했다. 청각의 가청 주파수는 약 15~30,000Hz(최대 45,000Hz 이상)로 인간(20~20,000Hz)보다 훨씬 넓은 범위를 인지한다.",
        "'강아지'는 '개'에 짐승의 새끼를 뜻하는 접미사 '-아지'가 붙은 형태이다. 이는 '말'에 '-아지'가 붙은 '망아지'와 동일한 단어 형성 원리를 따른다.",
        "강아지의 지능은 인간의 2~3세 유아와 비슷하다. 지능에 영향을 미치는 주요 요인으로는 품종(유전적 요인), 성장 환경, 그리고 사회화 교육이 있다.",
        "생후 3~12주 사이의 사회화 시기에 외부 자극이나 타인과의 교감이 부족할 경우, 성견이 되었을 때 낯선 대상에 대해 극심한 공포심이나 공격적 성향을 보일 수 있다.",
        "발바닥 패드는 지면의 충격을 흡수하고 발을 보호한다. 개는 땀샘이 발바닥에만 집중되어 있어 땀을 통한 체온 조절 능력이 낮으며, 이를 보완하기 위해 입으로 헐떡이며 열을 배출한다.",
        "일반적으로 소형견이 대형견보다 수명이 길다. 대형견은 신체 대사율이 높고 세포 분열 속도가 빨라 노화가 소형견보다 일찍 진행되는 경향이 있기 때문이다.",
        "초콜릿의 테오브로민 성분은 신경계 및 심장에 중독을 일으키며, 양파나 마늘에 포함된 성분은 강아지의 적혈구를 파괴하여 용혈성 빈혈을 유발한다.",
        "본래 개의 새끼를 뜻하는 단어였으나, 대상에 대한 친근함과 애정을 담아 부르는 애칭으로 쓰이면서 현대에는 성견을 포함한 반려견 전체를 일컫는 완곡한 표현으로 의미가 확장되었다."
    ]
    
    cfg_rag = {
        "title": "RAG Config",
        "model_name": "gpt-4o-mini", # "model_name": "gpt-4o",
        "embedding_model_name": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 100
    }
    print_cfg(cfg_rag)
    rag = RAGProcessor(file_path, cfg_rag)
    evaluator = RAGEvaluator(llm=rag.llm, embeddings=rag.embedding_model)

    # gpt-4o
    # {'faithfulness': 0.9333, 'answer_relevancy': 0.4736, 'context_recall': 0.8000, 'context_precision': 0.6917}

    # {'faithfulness': 0.7857, 'answer_relevancy': 0.5961, 'context_recall': 0.8000, 'context_precision': 0.9000}

    # 1. Baseline 파이프라인 실행 및 평가
    cfg = {
        "title": "Baseline", 
        "k": 3, 
        "use_multi_query": False,
        "search_type": "similarity",
        "search_kwargs": {"k": 3}
    }
    chain, retriever = rag.get_rag_chain(cfg)
    res = run_rag_pipeline(evaluator, chain, retriever, eval_questions, ground_truths, cfg)
    df = pd.DataFrame([res])
    print(df)


    # # 2. MMR 실험 실행 및 평가
    # cfg = {
    #     "title": "Improved + MMR", 
    #     "k": 3, 
    #     "use_multi_query": True, 
    #     "search_type": "mmr", 
    #     "search_kwargs": {"k": 3, "fetch_k": 10}
    # }
    # chain, retriever = rag.get_rag_chain(cfg)
    # res = run_rag_pipeline(evaluator, chain, retriever, eval_questions, ground_truths, cfg)
    # print(res)

    # # 2. Multi Query (Improved) 파이프라인 실행 및 평가
    # cfg = {
    #     "title": "Multi Query", 
    #     "k": 3, 
    #     "use_multi_query": True,
    #     "search_type": "similarity",
    #     "search_kwargs": {"k": 3}
    # }
    # chain, retriever = rag.get_rag_chain(cfg)
    # res = run_rag_pipeline(evaluator, chain, retriever, eval_questions, ground_truths, cfg)
    # print(res)



if __name__ == "__main__":
    main()