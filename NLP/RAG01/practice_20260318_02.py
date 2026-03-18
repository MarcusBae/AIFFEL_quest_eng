# RAG 시스템 비교 분석: Baseline vs Improved (Multi-Query)

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
from ragas.metrics.collections import (
    faithfulness, answer_relevancy, context_recall, context_precision,
)

class RAGProcessor:
    def __init__(self, file_path, model_name="gpt-4o", embedding_model_name="text-embedding-3-small", chunk_size=1000, chunk_overlap=100):
        load_dotenv()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        
        # 문서 로드 및 분할
        loader = TextLoader(file_path)
        pages = loader.load() # TextLoader doesn't have load_and_split usually, it loads as a single document by default or multiple if specified

        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
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

    def get_baseline_chain(self, k=3):
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain, retriever

    def get_improved_chain(self, k=3):
        base_retriever = self.db.as_retriever(search_kwargs={"k": k})
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )
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
            "question": [x["question"] for x in data_list],
            "answer": [x["answer"] for x in data_list],
            "contexts": [x["contexts"] for x in data_list],
            "ground_truth": [x["ground_truth"] for x in data_list]
        })
        return evaluate(
            ds, 
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=self.llm,
            embeddings=self.embeddings
        )

def main():
    nest_asyncio.apply()
    
    if "__file__" in globals():
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        ROOT_DIR = os.getcwd()
    
    file_path = os.path.join(ROOT_DIR, "data/dog_wiki.txt")
    rag = RAGProcessor(file_path)
    
    chain_b, retriever_b = rag.get_baseline_chain()
    chain_i, retriever_i = rag.get_improved_chain()

    # 1. 평가 질문 리스트
    eval_questions = [
        "강아지의 정의는 무엇인가요?",
        "강아지가 성체 개가 되기까지 보통 어느 정도의 시간이 걸리나요?",
        "강아지의 사회화 시기는 대략 언제부터 언제까지인가요?",
        "강아지에게 사회화 교육이 중요한 이유는 무엇인가요?",
        "갓 태어난 강아지의 신체적 특징 중 감각에 대한 설명은 무엇인가요?",
        "강아지가 젖을 떼고 사료를 먹기 시작하는 시기는 언제인가요?",
        "강아지에게 치명적일 수 있는 음식 중 대표적인 것은 무엇인가요?",
        "강아지의 예방 접종은 보통 언제부터 시작하나요?",
        "강아지라는 단어의 어원은 무엇인가요?",
        "강아지를 입양할 때 환경 조성에서 가장 주의해야 할 점은 무엇인가요?"
    ]

    # 2. 실제 정답(Ground Truth) 리스트
    # RAG 평가 시 실제 문서 내용과 생성된 답변을 비교하는 기준이 됩니다.
    ground_truths = [
        ["개의 새끼를 일컫는 말입니다."],
        ["품종에 따라 차이가 있으나 보통 1년 정도가 지나면 성체 개가 됩니다."],
        ["보통 태어난 후 3주부터 12주 사이를 사회화 시기라고 합니다."],
        ["이 시기에 겪는 경험이 성견이 되었을 때의 성격과 행동 양식에 큰 영향을 미치기 때문입니다."],
        ["눈을 뜨지 못해 앞을 볼 수 없으며, 귀도 막혀 있어 소리를 듣지 못합니다."],
        ["보통 생후 4주에서 6주 사이부터 젖을 떼고 이유식을 시작합니다."],
        ["초콜릿, 양파, 포도 등이 대표적인 위험 음식입니다."],
        ["모체로부터 받은 면역력이 떨어지는 생후 6~8주 경부터 시작하는 것이 일반적입니다."],
        ["'개'에 어린 것을 뜻하는 접미사 '-아지'가 붙어 만들어진 단어입니다."],
        ["강아지가 삼킬 수 있는 작은 물건을 치우고, 위험한 전선 등을 정리하여 안전한 환경을 만드는 것입니다."]
    ]
    
    print("--- Running Pipeline ---")
    results = []
    for q, gt in zip(eval_questions, ground_truths):
        # Baseline
        ans_b = chain_b.invoke(q)
        ctx_b = [doc.page_content for doc in retriever_b.invoke(q)]
        results.append({"question": q, "answer": ans_b, "contexts": ctx_b, "ground_truth": gt, "type": "baseline"})
        
        # Improved
        ans_i = chain_i.invoke(q)
        ctx_i = [doc.page_content for doc in retriever_i.invoke(q)]
        results.append({"question": q, "answer": ans_i, "contexts": ctx_i, "ground_truth": gt, "type": "improved"})

    evaluator = RAGEvaluator(llm=rag.llm, embeddings=rag.embedding_model)
    
    print("\n--- Evaluating Baseline ---")
    res_b = evaluator.run_eval([r for r in results if r["type"] == "baseline"])
    print(res_b)
    
    print("\n--- Evaluating Improved ---")
    res_i = evaluator.run_eval([r for r in results if r["type"] == "improved"])
    print(res_i)

if __name__ == "__main__":
    main()