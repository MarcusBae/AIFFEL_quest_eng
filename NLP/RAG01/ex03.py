# https://github.com/teddylee777/langchain-kr/tree/main



from langchain_openai import ChatOpenAI

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
        llm = ChatOpenAI(temperature=0.1)
        question = "대한민국의 수도는 ?"
        response = llm.invoke(question)
        print(response.content)
        print(response.response_metadata)

        # LogProb 활성화
        # 객체 생성
        llm_with_logprob = ChatOpenAI(
            temperature=0.1,  # 창의성 (0.0 ~ 2.0)
            max_tokens=2048,  # 최대 토큰수
            model_name="gpt-4.1-nano",  # 모델명
        ).bind(logprobs=True)

        response = llm_with_logprob.invoke(question)
        print(response.content)
        print(response.response_metadata)

    def run(self):
        self.func1()

if __name__ == "__main__":
    Chapt_01_Baseic().run()
