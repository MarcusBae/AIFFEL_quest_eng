import os
import random
import warnings
import requests
import pdfplumber
from fpdf import FPDF
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 경고 무시 및 환경 변수 로드
warnings.filterwarnings("ignore")

if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# --- 도구 정의 (Tools) ---

# @tool 
#   * decorator로 AI tool임을 명시
#   * 함수를 에이전트의 도구로 변환: 단순한 파이썬 함수를 AI 에이전트(Agent)가 실행할 수 있는 인터페이스로 변경
#   * 메타데이터 추출: AI는 함수의 이름, 설명(Docstring), 입력 파라미터 타입을 보고 "이 도구를 언제 써야 할지" 판단
#   * 스키마 자동 생성: AI 모델이 이해할 수 있는 JSON 형태의 스키마를 자동으로 생성하여, 모델이 어떤 인자를 넣어야 할지 명시
@tool 
def read_pdf(file_path: str):
    """
    PDF -> 텍스트 추출
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text.strip() else "ERROR: PDF에서 텍스트를 추출할 수 없습니다."
    except Exception as e:
        return f"ERROR: PDF 읽기 오류: {str(e)}"


# --- Pydantic 모델 정의 (Structured Output) ---
class HistoryChecker(BaseModel):
    """
    이전의 대화 기록을 참고하여 질문에 대해 답변할 수 있는지 판단.
    """
    yes_no: Literal["yes", "no"] = Field( 
        ...,        # Ellipsis: 이 필드는 필수값이다. 생략할 수 없다. (기존 파이썬에서는 생략 표시)
        description=("Use your previous conversation history to determine if you can answer your questions. "
                    "Return 'yes' if you can answer, 'no' if you can't answer.")
                    # description : msg to LLM 
    )

class AnswerChecker(BaseModel):
    """
    정답 분류기입니다.
    정답이 질문을 해결했는지 여부를 판단합니다.
    질문을 해결하지 못했을 시 해결될 때까지 도구를 이용합니다.
    질문을 해결했다면 "end", 해결하지 못했다면 "tool"을 반환합니다.
    """
    end: Literal["end", "tool"] = Field(
        ..., 
        description=("Determine if the correct answer has solved the question. "
                    "Return 'end' if you solved the question, or 'tool' if you didn't.")
    )

class State(TypedDict):
    query: Annotated[str, "User Question"]
    answer: Annotated[str, "LLM response"]
    messages: Annotated[list, add_messages] # 대화 이력, tool 실행 결과 이력 
    tool_call: Annotated[dict, "Tool Call Result"]

class ReportAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0) -> None:
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # 도구 설정
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 구조화된 출력기 설정
        self.history_checker_llm = self.llm.with_structured_output(HistoryChecker)
        self.answer_checker_llm = self.llm.with_structured_output(AnswerChecker)
        
        # 그래프 빌드 및 컴파일
        self.graph = self._build_graph()

    def _setup_tools(self) -> list:
        # write_pdf 도구 내부에 self.llm 접근이 필요하므로 여기서 정의하거나 클래스 메서드로 전달
        @tool
        def write_pdf(content: str, filename: str = "output.pdf", summary: bool = True):
            """
            텍스트를 PDF 파일로 저장하는 도구입니다.
            PDF형태의 문서로 만들어야할 때 이 도구를 사용하세요.
            """
            if summary:
                prompt = PromptTemplate.from_template("""
                        당신은 보고서를 작성하는 어시스턴트입니다. 당신에겐 문서 모음이 제공되고 이를 잘 분석하여 보고서를 작성하여야 합니다.
                        아래의 content는 문서 모음입니다. 문서의 제목, 본문을 잘 판단하고 정리하여 요약합니다.
                        항상 구조화된 출력을 제공하세요.
                        항상 마지막엔 인사이트도 첨부합니다.

                        content : {content}
                        """)

                chain = prompt | self.llm
                content = chain.invoke({"content": content}).content

            # 시스템 폰트 경로 확인 (Linux 기준)
            font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "./fonts/NotoSansKR-Regular.ttf"  # 기존 경로 유지 (백업)
            ]
            
            font_path = None
            for p in font_paths:
                if os.path.exists(p):
                    font_path = p
                    break
            
            if not font_path:
                print("Warring: 한국어 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            try:
                # fpdf2 uses fname instead of direct path, and supports UTF-8 by default
                pdf.add_font("NotoSans", style="", fname=font_path)
                pdf.set_font("NotoSans", size=12)
            except Exception as e:
                print(f"폰트 로드 실패 ({e}), 기본 폰트 사용")
                pdf.set_font("helvetica", size=12)

            for line in content.split("\n"):
                # 0 대신 pdf.epw (Effective Page Width) 사용
                pdf.multi_cell(pdf.epw, 10, line)
            
            output_path = os.path.join("./", filename)
            pdf.output(output_path)

            return f"{filename} 저장 완료"

        # 기타 도구들
        search_tool = None
        if os.getenv("TAVILY_API_KEY"):
            search_tool = TavilySearchResults(max_results=10)
        else:
            print("⚠️ TAVILY_API_KEY가 설정되지 않아 웹 검색 도구를 사용할 수 없습니다.")

        code_tool = PythonAstREPLTool()
        file_tools = FileManagementToolkit(selected_tools=["file_delete", "list_directory"]).get_tools()
        
        active_tools = [code_tool, write_pdf, read_pdf, *file_tools]
        if search_tool:
            active_tools.append(search_tool)
            
        return active_tools

    # --- 단기 기억 및 헬퍼 함수 ---
    def shorterm_memory(self, state: State) -> list:
        if len(state["messages"]) > 8:
            history = state["messages"][-8:-1]
        elif len(state["messages"]) == 1:
            history = ""
        else:
            history = state["messages"][:-1]
        return history

    # ------------------ 그래프 노드 -----------------------------------------------
    # [Node] 초기 상태 설정
    def history_node(self, state: State) -> dict:
        if len(state["messages"]) == 1: # 아예 최초 시작 시
            return {"answer": "답변 없음", "tool_call": "사용된 도구 없음"}
        return state
        
    # [Node] 이전 대화 기록 만으로 현재 질문에 답변 가능한지 판단하는 조건부 노드
    def history_check_node(self, state: State) -> str:
        prompt = PromptTemplate.from_template("""
                이전의 대화 기록을 참고하여 질문에 대해 답변할 수 있는지 판단합니다.
                답변할 수 있다면 "yes", 답변할 수 없다면 "no"를 반환합니다.

                대화 기록 : {history}
                질문 : {query}
                """)
        
        chain = prompt | self.history_checker_llm
        history = self.shorterm_memory(state)
        result = chain.invoke({
            "history": history,
            "query": state["query"]
        })
        return result.yes_no

    # [Node] memory_chat: 이전 대화 기록(history)을 기반으로 직접 답변을 생성하는 노드
    def memory_chat(self, state: State) -> dict:
        prompt = PromptTemplate.from_template("""
                이전의 대화 기록을 참고하여 질문에 대해 답변하세요.
                아래 대화 기록을 첨부합니다.
                대화 기록을 통해 답변이 어렵다면 내부 지식을 참조하세요.

                대화 기록 : {history}
                질문 : {query}
                """)
        
        chain = prompt | self.llm
        history = self.shorterm_memory(state)
        result = chain.invoke({
            "history": history, 
            "query": state["query"]
        })

        if not state.get("tool_call"):
            return {"answer": result.content, "messages": [result], "tool_call": "사용된 기록 없음."}
        else:
            return {"answer": result.content, "messages": [result]}

    # [Node] 질문 해결을 위한 도구 (Tool) 선택
    def select_node(self, state: State) -> dict:
        prompt = PromptTemplate.from_template("""
                이전의 대화 기록을 참고하여 질문에 대해 답변하세요.
                아래 대화 기록을 첨부합니다.
                이전의 대화가 다음에 어떤 도구를 사용해야하는지 힌트가 될 수 있습니다. 꼭 참조하세요.
                도구의 변화가 큰 결과를 가져올 수 있습니다.
                들어온 메시지, 정답, 이전 기록을 모두 분석하여 가장 적절한 도구를 선택하세요.

                대화 기록 : {history}
                최근 사용한 도구 : {tool_name}
                정답 : {answer}
                질문 : {query}
                """)
        
        chain = prompt | self.llm_with_tools
        history = self.shorterm_memory(state)
        result = chain.invoke({
            "history": history,
            "tool_name": state.get("tool_call", "없음"),
            "answer": state.get("answer", "없음"),
            "query": state["query"]
        })

        if hasattr(result, "tool_calls") and len(result.tool_calls) > 0:
            return {"messages": [result], "tool_call": result.tool_calls}
        else:
            # 도구를 선택하지 못한 경우, response 노드로 우회하도록 유도 (ToolNode 방지)
            return {
                "messages": [AIMessage(content="도구를 선택하지 못했습니다. 질문에 직접 답변하거나 다른 방향으로 시도하세요.")],
                "tool_call": "선택된 도구 없음"
            }

    # [Node] response_node: 생성된 마지막 메시지에서 최종 답변 내용을 추출하여 answer 상태에 저장하는 노드
    def response_node(self, state: State) -> dict:
        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {"answer": content}

    # [Node] answer_check_node: 생성된 답변이 사용자의 질문을 해결했는지 확인하여 종료 여부를 결정하는 조건부 노드
    def answer_check_node(self, state: State) -> str:
        prompt = PromptTemplate.from_template("""
        당신은 정답 분류기 어시스턴트입니다.
        정답이 질문을 해결하였는지 여부를 판단합니다.
        질문을 해결하지 못했다면 도구를 이용합니다.
        질문을 해결하였다면 "end", 아니라면 "tool"을 반환합니다.
        기존 History도 참고하여 답변하세요.

        History : {history}
        정답 : {answer}
        질문 : {query}
        """)

        chain = prompt | self.answer_checker_llm
        history = self.shorterm_memory(state)
        result = chain.invoke({
            "history": history,
            "answer": state["answer"],
            "query": state["query"]
        })
        return result.end

    # --- 그래프 구축 ---
    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(State)
        
        builder.add_node("history_node", self.history_node)
        builder.add_node("memory_chat", self.memory_chat)
        builder.add_node("select", self.select_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("response", self.response_node)

        builder.add_edge(START, "history_node")
        builder.add_conditional_edges(
            "history_node",                 # 출발지 
            self.history_check_node,        # 판단해서 
            {"yes": "memory_chat", "no": "select"}  # 갈림길
        )
        
        # select에서 도구가 선택되지 않았을 때의 처리 (conditional_edge로 변경 권장되나 일단 edge 기반 수정)
        def route_after_select(state: State):
            if state.get("tool_call") == "선택된 도구 없음":
                return "response"
            return "tools"

        builder.add_conditional_edges(
                "select", 
                route_after_select, 
                {"tools": "tools", "response": "response"})

        builder.add_edge("tools", "response")
        builder.add_edge("memory_chat", "response")
        builder.add_conditional_edges(
            "response",
            self.answer_check_node,
            {"end": END, "tool": "select"}
        )

        memory = MemorySaver()

        # builder.compile 
        #   컴파일 필요한 이유
        #   유효성 검사: 정상 그래프 확인
        #   영속성 연결: 데이터베이스(Checkpointer)를 그래프에 바인딩하여, 
        #       실행할 때마다 별도의 설정 없이도 자동으로 대화 기록이 저장되라고 
        return builder.compile(checkpointer=memory)

    def streaming(self, query: str, config: RunnableConfig, mode: str = "values") -> None:
        result = self.graph.stream( # PregelInvoke Generator, Iterator                          
            {"messages": [("user", query)], "query": query}, 
            config=config, 
            stream_mode=mode
        )

        if mode == "values":
            for step in result: # 결과가 하나씩 생성. Lazy Execution, Generation 
                if "messages" in step:
                    messages = step["messages"]
                    if messages:
                        messages[-1].pretty_print()
        elif mode == "updates":
            for step in result:
                for k, v in step.items():
                    print(f"\n\n=== {k} ===\n\n")
                    print(v)

# --- 메인 실행부 ---
if __name__ == "__main__":
    agent = ReportAgent()
    
    # 설정 초기화
    thread_id = random.randint(1, 999999)
    config = RunnableConfig(
            recursion_limit=20,
            configurable={"thread_id": thread_id}
                        # thread_id: 특정 대화 세션을 식별하는 고유 번호, checkpoint에서 load 
    )
    print("-" * 30)
    agent.streaming("1+1 은?", config)

    # print("-" * 30)
    # print("\n\n===== 에이전트 시작: 현대자동차 보고서 요청 =====")
    # # 실제 실행 시 API 호출 및 검색이 발생함
    # query = "현대자동차라는 회사에 대해 조사해주세요. 잘 정리된 보고서를 제공해주십시오. pdf 포멧의 파일로 받기를 희망합니다."
    
    # agent.streaming(query, config)
    # print("(검색 및 PDF 생성이 포함되어 있어 주석 처리함. 필요 시 해제하여 사용)")
