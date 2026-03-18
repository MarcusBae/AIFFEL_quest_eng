# https://github.com/teddylee777/langchain-kr/tree/main



from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def print_func_name(func):
    def wrapper(*args, **kwargs):
        print("-" * 60)
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper

class Chapt_01_Baseic():
    def __init__(self):
        pass

    @print_func_name
    def func1(self):
        # ChatOpenAI
        llm = ChatOpenAI(temperature=0.1)  # 타입: ChatOpenAI (랭체인의 모델 객체)
        question = "대한민국의 수도는 ?"   # 타입: str (문자열)
        
        response = llm.invoke(question)    # 타입: AIMessage (AI의 응답을 담은 객체)
        print(response.content)
        print(response.response_metadata)

        # LogProb 활성화
        # 객체 생성
        llm_with_logprob = ChatOpenAI(     # 타입: RunnableBinding (옵션이 바인딩된 실행 객체)
            temperature=0.1,  # 창의성 (0.0 ~ 2.0)
            max_tokens=2048,  # 최대 토큰수
            model_name="gpt-4.1-nano",  # 모델명
        ).bind(logprobs=True)

        response = llm_with_logprob.invoke(question)  # 타입: AIMessage
        print(response.content)
        print(response.response_metadata)

    def run(self):
        self.func1()


class Chapt_02_Prompt():
    def __init__(self):
        pass
    @print_func_name
    def func1(self):
        llm = ChatOpenAI(temperature=0.1)

        template = "{country}의 수도는 ?"
        prompt_template = PromptTemplate.from_template(template)

        chain = prompt_template | llm                             # RunnableSequence 

        formatted_prompt = prompt_template.format(country='대한민국')

        r = chain.invoke({"country": "대한민국"}).content
        # r = llm.invoke(formatted_prompt).content
        print(r)

    @print_func_name
    def func2(self):
        llm = ChatOpenAI(temperature=0.1)
        template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["country1"],
            partial_variables={
                "country2": "미국"  # dictionary 형태로 partial_variables를 전달
            },
        )

        chain = prompt_template | llm

        r = chain.invoke({"country1": "대한민국"}).content
        print(r)

        formatted_prompt = prompt_template.format(country1="대한민국")
        r = llm.invoke(formatted_prompt).content
        print(r)

    @print_func_name
    def func3(self):
        pass

    @print_func_name
    def func4_few_shot_prompt_template(self):
        llm = ChatOpenAI(temperature=0.1)
        question = "대한민국의 수도는 무엇이니?"

        answer = llm.stream(question)
        # stream_response(answer)
        # skip...

    @print_func_name
    def func5_prompt_from_hub(self):
        # from langchain import hub
        #import langchainhub as hub
        from langchain_classic import hub
        
        prompt = hub.pull('rlm/rag-prompt')
        print(prompt)
    
    def run(self):
        self.func5_prompt_from_hub()

if __name__ == "__main__":
    # Chapt_01_Baseic().run()
    Chapt_02_Prompt().run()