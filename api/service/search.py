import os
from data.pinecone import search as search
from model.airesults import AIResults
from model.resource import Resource
from .adaptive_retrieval import AdaptiveRAG
from langchain_core.runnables import  RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from .self_rag import self_rag as self_rag
from .crag import crag as crag
from config import config as config

llm_model_name = config.get_llm_model_config()['model_name']

adaptive_query_engine = AdaptiveRAG()

def get_query(query:str)-> list[Resource]:
    resources, _ = search.similarity_search(query)
    return resources

def do_self_rag(query:str, top_k) -> str:
    response = self_rag.self_rag(query=query, top_k = top_k)
    print("Response : ", response)
    default_text = f"""Result of Self-RAG: \n\n"""
    return AIResults(text = default_text + response, ResourceCollection=[]) 

def do_crag(query:str, k:int) -> str:
    response = crag.crag_process(query=query, k = k)
    print("Response : ", response)
    default_text = f"""Result of CRAG: \n\n"""
    return AIResults(text = default_text + response, ResourceCollection=[]) 

def get_adaptive_query(query:str, k:int = 3, rerank_mode: bool = True, query_category = "Auto") -> str:
    response, resources = adaptive_query_engine.answer(query, k, rerank_mode, query_category)
    print("Response : ", response)
    print("resources : ", len(resources))
    default_text = f"""Rerank_mode : {rerank_mode}, query_category : {query_category} \n\n"""
    return AIResults(text = default_text + response, ResourceCollection=resources) 

def get_llm_response(query:str) -> str:
    template = """
    Answer the question. If you can't 
    answer the question, reply "I don't know".
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm_model = GoogleGenerativeAI(model=llm_model_name)

    rag_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
    )
    default_text = "This question is not related to the book !! This is the answer based on my knowledge :\n\n"
    return AIResults(text=default_text + rag_chain.invoke(query),ResourceCollection=[])