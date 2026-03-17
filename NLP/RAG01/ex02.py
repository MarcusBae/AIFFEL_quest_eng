#------------------------------------------------------------------------------
# Rag From Scratch: Overview
#
#   https://github.com/langchain-ai/rag-from-scratch/tree/main
#------------------------------------------------------------------------------

import bs4
import functools
import numpy as np
import os
import tiktoken
from dotenv import load_dotenv
from operator import itemgetter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#------------------------------------------------------------------------------
# 메모
#   import 에러가 계속 난다. 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 0. Environment
#------------------------------------------------------------------------------

# pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain

#------------------------------------------------------------------------------
# 0. Utils
#------------------------------------------------------------------------------

def print_func_name(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n" + "-"*20 + f" {func.__name__}() " + "-"*20)
        return func(*args, **kwargs)
    return wrapper


# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = <your-api-key>



#------------------------------------------------------------------------------
# Part 1: Overview
#------------------------------------------------------------------------------
@print_func_name
def part_01_overview():
    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    #### RETRIEVAL and GENERATION ####

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\nQuestion: {question} \nContext: {context} \nAnswer:"
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    rag_chain.invoke("What is Task Decomposition?")

#------------------------------------------------------------------------------
# Part 2: Indexing
#------------------------------------------------------------------------------
@print_func_name
def part_02_indexing():
    # Documents
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    r = num_tokens_from_string(question, "cl100k_base")
    print(f"num_tokens_from_string: {r} {question}")
    
    embd = OpenAIEmbeddings()
    
    query_result = embd.embed_query(question)    # 1536 차원
    document_result = embd.embed_query(document)

    print(f"len(query_result): {len(query_result)}")
    print(f"len(document_result): {len(document_result)}")


    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    similarity = cosine_similarity(query_result, document_result)
    print("Cosine Similarity:", similarity)

    # Load blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    # Index
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

#------------------------------------------------------------------------------
# Part 3: Retrieval
#------------------------------------------------------------------------------
@print_func_name
def part_03_retrieval():
    pass

#------------------------------------------------------------------------------
# Part 4: Generation
#------------------------------------------------------------------------------
@print_func_name
def part_04_generation():
    pass    

#------------------------------------------------------------------------------
# Part 5: Multi Query
# https://docs.langchain.com/oss/python/langchain/overview
#------------------------------------------------------------------------------
@print_func_name
def part_05_multi_query():

    # 1. INDEXING --------------------------------------------------------- 
    # Load blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    # Index
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    # 2. Prompt  --------------------------------------------------------- 
    # from langchain.prompts import ChatPromptTemplate

    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    # LCEL(LangChain Expression Language) : 유닉스 파이프라인(|) 기법, 좌 -> 우로 순처 처리 
    generate_queries = (
        prompt_perspectives             # 입력 
        | ChatOpenAI(temperature=0)     # 모델
        | StrOutputParser()             # 모델 출력 중 문자열만 추출
        | (lambda x: x.split("\n"))     # 문자열 -> list
    )

    # ??? Multi Query도 LLM으로 생성했다. local, remote LLM 말고 다른 방법이 또 있을까? 
    query = "What is Task Decomposition?"
    generated_queries = generate_queries.invoke({"question": query})

    print(f"Original Query: {query}")
    for new_query in generated_queries:
        print(f"Generated Query: {new_query}")


    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    len(docs)


    # RAG
    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"question":question})

#------------------------------------------------------------------------------
# Part 6: RAG Fusion
#------------------------------------------------------------------------------
@print_func_name
def part_06_rag_fusion():
    # 1. INDEXING --------------------------------------------------------- 
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    blog_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(blog_docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results
    
    question = "What is task decomposition for LLM agents?"
    llm = ChatOpenAI(temperature=0)

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    r = len(docs)
    print(r)

    # RAG
    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    r =  final_rag_chain.invoke({"question":question})
    print(r)

#------------------------------------------------------------------------------
# Part 9: HyDE
#------------------------------------------------------------------------------
@print_func_name
def part_09_test_hyde():
    
    # 1. INDEXING --------------------------------------------------------- 
    # Load blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    blog_docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    # Index
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 2. RETRIEVAL --------------------------------------------------------- 
    # 가상 문서 생성 : HyDE document generation
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
    )

    # Run
    question = "What is task decomposition for LLM agents?"
    # generate_docs_for_retrieval.invoke({"question":question})

    # Retrieve Chain 구성 
    retrieval_chain = generate_docs_for_retrieval | retriever 
    retrieved_docs = retrieval_chain.invoke({"question":question})
    
    print(type(retrieved_docs))
    print(dir(retrieved_docs))
    print(len(retrieved_docs))
    print(retrieved_docs[0])

    # 3. GENERATION ---------------------------------------------------------
    # 최종 답변을 위한 프롬프트 템플릿
    # 검색된 문서(context)와 사용자의 질문(question)을 결합합니다.
    template_generation = """Answer the following question based on the provided context. 
    If the answer is not in the context, say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""

    prompt_gen = ChatPromptTemplate.from_template(template_generation)

    # 문서들을 하나의 문자열로 합치는 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 전체 RAG 체인 구성
    # 1. HyDE로 검색된 문서들(retrieved_docs)을 가져옴
    # 2. 질문과 검색된 문서를 프롬프트에 주입
    # 3. LLM이 최종 답변 생성
    rag_chain = (
        {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
        | prompt_gen
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
    )

    print("\n=== Final Answer ===")
    final_answer = rag_chain.invoke(question)
    print(final_answer)

def test():
    from importlib.metadata import version

    print("[LangChain Package Version]")
    for package_name in [
        "langchain",
        "langchain-core",
        "langchain-experimental",
        "langchain-community",
        "langchain-openai",
        "langchain-teddynote",
        "langchain-huggingface",
        "langchain-google-genai",
        "langchain-anthropic",
        "langchain-cohere",
        "langchain-chroma",
        "langchain-elasticsearch",
        "langchain-upstage",
        "langchain-cohere",
        "langchain-milvus",
        "langchain-text-splitters",
    ]:
        try:
            package_version = version(package_name)
            print(f"{package_name}: {package_version}")
        except ImportError:
            print(f"{package_name}: 설치되지 않음")


test()
# part_09_test_hyde()



