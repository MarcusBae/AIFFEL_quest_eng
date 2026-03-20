from langgraph.graph import StateGraph, START, END
from typing import TypedDict

def part_1_1_LangGraph_001):
    # 1단계: 데이터 저장소 (State) - 기억해야 할 정보
    class MyState(TypedDict):
        message: str

    # 2단계: 작업 함수 만들기 (Node) - old state -> new state 
    def say_hello(state):
        return {"message": "Hello, LangGraph!"}

    # 3단계: 그래프 만들기
    graph = StateGraph(MyState)
    graph.add_node("hello", say_hello)
    graph.add_edge(START, "hello")
    graph.add_edge("hello", END)

    # 4단계: 실행하기
    app = graph.compile()   # 그래프를 실행 가능한 프로그램으로 생성
    result = app.invoke({"message": "1st"})
    print(result)

def part_1_1_LangGraph_002():
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    # 1️⃣ State: 카운터를 저장하는 상자
    class CounterState(TypedDict):
        count: int

    # 2️⃣ Node: 카운터를 증가시키는 함수
    def increment(state):
        print(f"현재 카운트: {state['count']}")
        new_count = state["count"] + 1
        print(f"새로운 카운트: {new_count}")
        return {"count": new_count}

    # 3️⃣ Edge: 노드들을 연결하는 그래프
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)

    # 실행해보기
    app = graph.compile()
    result = app.invoke({"count": 0})
    print(f"최종 결과: {result}")

def part_1_1_LangGraph_003():
    # 1️⃣ State: 카운터를 저장하는 상자
    class CounterState(TypedDict):
        count: int

    # 첫 번째 증가 함수
    def first_increment(state):
        print("첫 번째 증가")
        return {"count": state["count"] + 1}

    # 두 번째 증가 함수  
    def second_increment(state):
        print("두 번째 증가")
        return {"count": state["count"] + 10}

    # 그래프 구성
    graph = StateGraph(CounterState)
    graph.add_node("first", first_increment)
    graph.add_node("second", second_increment)

    # 연결: START → first → second → END
    graph.add_edge(START, "first")
    graph.add_edge("first", "second") 
    graph.add_edge("second", END)

    # 실행
    app = graph.compile()
    result = app.invoke({"count": 0})
    print(f"최종 결과: {result}")

def part_1_1_LangGraph_004():
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages

    # add_messages
    #   상태 필드에 리듀서(Reducer)를 지정하면 
    #   노드가 반환한 값을 기존 상태에 어떻게 적용할지 정의할 수 있습니다. 
    #   특히 대화형 앱에서는 add_messages 리듀서가 표준 패턴입니다.
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]     # Annotated[실제 타입, 메타데이터] 형식의 type hinting 기능
                                                    # LangGraph는 메터 데이터를 읽고 상태 업데이트 방식을 결정
        user_name: str                              # 일반 필드 (덮어쓰기)

    from langgraph.graph import MessagesState
    class MyState(MessagesState): # 메시지 히스토리가 필요하면 MessagesState를 상속
        # messages 필드 + add_messages 리듀서가 자동 포함
        user_name: str

def part_1_3_StateGraph():
    # 상태 관리의 중요성
    class ChatState(TypedDict):
        user_message: str      # 현재 사용자 메시지
        chat_history: list     # 이전 대화 기록
        user_context: dict     # 사용자 정보
        system_status: str     # 시스템 상태

    # 최근 대화 맥락 사용해서 
    def conversation_node(state: ChatState):                # 시그니처: 대 부분 state + config (optional)
        context = "\n".join(state["chat_history"][-3:])     # context 추출
        response = generate_contextual_response(state["user_message"], context)
        return {"ai_response": response}                    # 부분 업데이트 

    from typing_extensions import TypedDict
    from typing import Annotated, List
    from operator import add

    class BasicState(TypedDict):
        # 단순 값들 - 덮어쓰기
        current_step: str
        user_id: str

        # 누적되는 값들 - 추가
        messages: Annotated[List[str], add]
        processing_log: Annotated[List[str], add]

        # 리듀서(Reducer)란?
        #   함수형 프로그래밍 개념 - "기존 값과 새 값을 어떻게 합칠 것인가"를 정의하는 함수

def part_1_3_StateSchema():
    # 1. TypedDict
    from typing_extensions import TypedDict, NotRequired
    from typing import Annotated
    from operator import add

    class MyState(TypedDict):
        query: str                          # 필수 필드
        results: Annotated[list[str], add]  # 필수 (리듀서 적용)
        count: NotRequired[int]             # 선택적 필드 (노드에서 생략 가능)
        
    # DataClass
    from dataclasses import dataclass, field
    from typing import Annotated
    from operator import add

    # 2. dataclass - 기본값이 필요할 때 유용
    @dataclass
    class MyState:
        query: str = ""
        results: Annotated[list[str], add] = field(default_factory=list)
        count: int = 0

    # 3. Pydantic BaseModel
    from pydantic import BaseModel, Field
    from typing import Annotated
    from operator import add

    class MyState(BaseModel):
        query: str = ""
        results: Annotated[list[str], add] = Field(default_factory=list)
        count: int = Field(default=0, ge=0)  # 0 이상만 허용

def part_1_1_LangGraph_005():

    # 스키마 TYPE 1 - Single Schema - 입출력이 동일
    from typing_extensions import TypedDict
    from typing import Annotated
    from operator import add

    class BasicState(TypedDict):
        user_input: str
        ai_response: str
        conversation_history: Annotated[list[str], add]

    def chatbot_node(state: BasicState) -> dict:
        response = f"'{state['user_input']}'에 대한 응답입니다."
        return {
            "ai_response": response,
            "conversation_history": [f"User: {state['user_input']}", f"AI: {response}"]
        }

    # 스키마 TYPE 2. 명시적 입출력 스키마 (Explicit Input/Output Schema)
    class InputState(TypedDict):
        question: str

    class OutputState(TypedDict):
        answer: str

    class OverallState(InputState, OutputState):
        intermediate_data: str
        search_results: list[str]

    def search_node(state: InputState) -> dict:
        return {
            "search_results": ["결과1", "결과2"],
            "intermediate_data": f"'{state['question']}' 검색 완료"
        }

    def answer_node(state: OverallState) -> OutputState:
        return {"answer": f"검색 결과: {state['search_results'][0]}"}

def part_1_7_Functional_API_001():
    # Graph API 방식 - 간단한 작업에도 많은 boilerplate 코드 필요
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.checkpoint.memory import InMemorySaver

    def call_model(state: MessagesState):
        # 모델 호출 로직
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    # 그래프 정의
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)

    # 컴파일
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 실행
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "1"}}
    )

def part_1_7_Functional_API_001():
    # Functional API 방식 - 간결하고 직관적으로 개선
    from langgraph.func import entrypoint
    from langgraph.checkpoint.memory import InMemorySaver

    @entrypoint(checkpointer=InMemorySaver())
    def chat(messages: list):
        response = model.invoke(messages)
        return [response]

    # 실행
    result = chat(
        [{"role": "user", "content": "Hello"}],
        config={"configurable": {"thread_id": "1"}}
    )




