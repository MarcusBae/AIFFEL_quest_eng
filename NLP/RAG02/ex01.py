#------------------------------------------------------------------------------
# LangGraph
#   LangGraph로 데이터 분석을 수행하는 에이전트 기반 시스템
#   복잡한 로직을 **그래프(Graph)**로 표현합니다.

# Nodes (노드): 실제 작업을 수행하는 함수나 단계입니다. (예: "검색하기", "답변 생성하기", "코드 실행하기")
# Edges (엣지): 노드 사이의 연결 통로입니다. 다음으로 어떤 노드를 실행할지 결정하며, 조건에 따라 분기되는 '조건부 엣지'도 가능합니다.
# State (상태): 그래프 전체에서 공유되는 데이터 주머니입니다. 각 노드는 이 상태를 읽고 수정하며 작업을 이어갑니다.

import pandas as pd
import re
import koreanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import warnings
from typing import Annotated, Literal, Tuple
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")


load_dotenv()

# 그래프 상태 정의 : 그래프가 공유할 데이터 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]             # 대화 기록
    code: Annotated[str, "Python code"]                 # 생성된 파이썬 코드
    code_result: Annotated[str, "Python code Result"]   # 코드 실행 결과

class HistoryChecker(BaseModel):
    """이전 대화 기록을 참고하여 현재 질문에 답변 가능한지 판단"""
    yes_no: Literal["yes", "no"] = Field(
        ..., 
        description="Use history to determine if you can answer. 'yes' if possible, 'no' if not."
    )

# 데이터 분석 에이전트 
class DataAgent:
    def __init__(self, data_path: str, model: str = "gpt-4o-mini"):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.llm = ChatOpenAI(model=model, temperature=0.0)
        
        # 제목 및 요약 생성
        self.title, self.summary = self._create_title_summary()
        print(f"===== 데이터셋 분석 완료: {self.title} =====")
        
        # 도구 및 그래프 초기화
        self.tool = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools([self.tool])
        self.history_checker = self.llm.with_structured_output(HistoryChecker)
        self.graph = self._build_graph()

    def _setup_tools(self):
        return PythonAstREPLTool(   # 파이썬 코드 실행 도구  수 있는 도구(
            name="python_repl_ast",
            description="A Python shell. Use this to execute python commands. Input must be valid python code.",
            locals={"df": self.df}
        )

    def _create_title_summary(self) -> Tuple[str, str]:
        df_sampled = self.df.sample(n=3000) if len(self.df) > 3000 else self.df
        prompt = PromptTemplate.from_template("""
            당신은 요약 전문가입니다.
            데이터셋 : {df}
            데이터셋의 정보를 보고 제목과 요약을 만들어냅니다.
            
            제목:
            요약:
        """)
        chain = prompt | self.llm
        result = chain.invoke({"df": df_sampled})
        
        title, summary = "Untitled", "No Summary"
        try:
            content = result.content.replace("## 결과:\n\n", "")
            lines = content.split("\n")
            title = lines[0].replace("## 제목:", "").strip()
            summary = "\n".join(lines[1:]).replace("## 요약:", "").strip()
        except:
            pass
        return title, summary

    # --- Graph Nodes ---
    def history_node(self, state: State):
        return

    def history_check(self, state: State):
        if len(state["messages"]) <= 1:
            return "no"
        
        prompt = PromptTemplate.from_template("""
            이전 대화 기록을 참고하여 질문에 대해 답변할 수 있는지 판단합니다.
            대화 기록 : {history}
            질문 : {query}
        """)
        chain = prompt | self.history_checker
        result = chain.invoke({
            "history": state["messages"][:-1],
            "query": state["messages"][-1]
        })
        return result.yes_no

    def select(self, state: State):
        prompt = PromptTemplate.from_template("""
            당신은 데이터 분석가입니다. 데이터프레임 df를 활용해 코드를 작성하세요.
            df : {df}
            title : {title}
            summary : {summary}
            query : {query}
            
            'python_repl_ast' 도구를 사용해 질문에 대한 코드를 생성하세요.
            코드에 한글이 필요하다면 `import koreanize_matplotlib`을 사용하세요.
        """)
        chain = prompt | self.llm_with_tools
        result = chain.invoke({
            "df": self.df.head().to_string(),
            "title": self.title,
            "summary": self.summary,
            "query": state["messages"][-1].content
        })

        if hasattr(result, "tool_calls") and len(result.tool_calls) > 0:
            return {"code": result.tool_calls[0]["args"]["query"]}
        return {"code": ""}

    def code_executor(self, state: State):
        code = state["code"]
        if not code:
            return {"code_result": ""}
            
        try:
            if "plt" in code or "sns" in code:
                save_fig_code = """
import io
import base64
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
print(base64.b64encode(buf.read()).decode("utf-8"))
"""
                result = self.tool.invoke(code + save_fig_code)
                return {"code_result": result}
            else:
                result = self.tool.invoke(code)
                return {"code_result": str(result)}
        except:
            return {"code_result": ""}

    def code_response(self, state: State):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """코드 : {code} \n 결과 : {code_result}
            주어진 코드와 결과를 바탕으로 질의에 답변하세요. 프로그래밍 용어가 아닌 일반적인 설명과 인사이트를 제공하세요."""),
            ("human", "{query}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({
            "code": state["code"],
            "code_result": state["code_result"],
            "query": state["messages"][-1]
        })
        return {"messages": [result]}

    def general_response(self, state: State):
        prompt = PromptTemplate.from_template("""
            이전 대화 기록을 참고하거나 내부 지식을 참조하여 답변하세요.
            대화 기록 : {history}
            질문 : {query}
        """)
        chain = prompt | self.llm
        result = chain.invoke({
            "history": state["messages"][:-1],
            "query": state["messages"][-1]
        })
        return {"messages": [result]}

    def _build_graph(self) -> CompiledStateGraph: 
        builder = StateGraph(State)
        
        builder.add_node("history_node", self.history_node) # 질문이 "이전 대화"에 관한 것인지 확인합니다. (분기점)
        builder.add_node("select", self.select)     # (df) 정보를 바탕으로 분석에 필요한 Python 코드를 LLM이 생성
        builder.add_node("code_executor", self.code_executor)   # code 실행 
        builder.add_node("code_response", self.code_response)   # 응답
        builder.add_node("response", self.general_response)
        
        builder.add_edge(START, "history_node")
        builder.add_conditional_edges(
            "history_node",
            self.history_check,
            {"no": "select", "yes": "response"}
        )
        builder.add_edge("select", "code_executor")
        builder.add_edge("code_executor", "code_response")
        builder.add_edge("code_response", END)
        builder.add_edge("response", END)
        
        memory = InMemorySaver()
        return builder.compile(checkpointer=memory)

    # 출력 함수 정의
    # mode = "values" : 상태의 키, 값의 형태로 반환
    # mode = "updates" : 업데이트되는 값만 반환
    def streaming(self, query: str, config: RunnableConfig, mode: str = "values"):
        result = self.graph.stream(                 # result = Iterator (또는 Generator)
                {"messages": [("user", query)]}, 
                config=config, stream_mode=mode)
        
        if mode == "values":
            for step in result:
                if "messages" in step:
                    # 마지막 메시지만 출력
                    step["messages"][-1].pretty_print()
                    print("\n")
        elif mode == "updates":
            for step in result:
                for k, v in step.items():
                    print(f"\n=== {k} ===\n{v}")

if __name__ == "__main__":

    if "__file__" in globals():
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        ROOT_DIR = os.getcwd()
    
    # 1. 타이타닉 데이터 에이전트
    titanic_agent = DataAgent(os.path.join(ROOT_DIR, "data/titanic.csv"))

    # 에이전트 실행
    #   질문 -> 전체 워크플로우 실행, 실시간 결과 스트리밍
    #   RunnableConfig로 실행 제한(recursion_limit)을 설정하여 무한 루프를 방지합니다.
    config_titanic = RunnableConfig(recursion_limit=10, configurable={"thread_id": "titanic_1"})
    
    query = "생존자 비율 시각화하고 인사이트 제공해줘"
    titanic_agent.streaming(query, config_titanic)
    
    query = "아까 내가 질문했던 내용 다시 알려줘"
    titanic_agent.streaming(query, config_titanic)
    
    query = "아까 물어봤던 숫자들 다 더하면 몇인지 알려줘"
    titanic_agent.streaming(query, config_titanic)

    print("\n" + "="*50 + "\n")

    # # 2. 올림픽 데이터 에이전트
    # olympic_agent = DataAgent(os.path.join(ROOT_DIR, "data/athlete_events.csv"))
    # config_olympic = RunnableConfig(recursion_limit=10, configurable={"thread_id": "olympic_1"})
    
    # query = "올림픽에는 몇개의 나라가 출전했나요?"
    # olympic_agent.streaming(query, config_olympic)
    
    # query = "가장 많이 출전한 나라는 어디인가요?"
    # olympic_agent.streaming(query, config_olympic)
    
    # query = "올림픽에 출전한 선수들의 평균 체중은 얼마인가요?"
    # olympic_agent.streaming(query, config_olympic)
    
    # query = "키와 메달 획득과의 상관관계를 보여주세요. regplot 그래프로 그려주세요."
    # olympic_agent.streaming(query, config_olympic)
    
    # query = "키와 체중, 그리고 메달 획득과의 상관관계를 보여주세요. 산점도 그래프로 그려주세요."
    # olympic_agent.streaming(query, config_olympic)
    
    # query = ("키와 메달 획득과의 상관관계를 보여주세요."
    #         " boxplot 그래프로 그려주세요. 그리고 그래프에서 이상치로 보이는 선수들의 목록을" 
    #         " z-score나 IQR 방법을 사용해서 찾아서 표로 보여주세요.")
    # olympic_agent.streaming(query, config_olympic)
