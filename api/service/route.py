from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Tuple


class relation_check(BaseModel):  
    # setup: str = Field(description="Original query")
    check: bool  = Field(description="Is the query relevant?", )

def routing_query(query : str) -> bool:
    template = """You are a helpful assistant to check if the question is related to focusing on how to influence others focusing on improving interpersonal relationships by being genuinely interested in others, 
    listening attentively and showing genuine appreciation.
    
    The question: {question} \n
    Output : True or False"""
    prompt= ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)
    structured_llm = llm.with_structured_output(relation_check)
    checking_chain = (
        prompt 
        | structured_llm
    )
    result = checking_chain.invoke({"question" : query})
    return result.check
    