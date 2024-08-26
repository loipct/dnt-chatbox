import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)
llm1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)

class RetrievalResponse(BaseModel):
    response: str = Field(..., title="""Determine whether the content in the book "How to Win Friends and Influence People" can answer the query""", description="Output only 'Yes' or 'No'.")
retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Given the query '{query}', determine whether the content in the book "How to Win Friends and Influence People" can answer the query. Output only 'Yes' or 'No'."""
)

class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant", description="Output only 'Relevant' or 'Irrelevant'.")
relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")
generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported", description="Output 'Fully supported', 'Partially supported', or 'No support'.")
support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")
utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

# Create LLMChains for each step
retrieval_chain = retrieval_prompt | llm1.with_structured_output(RetrievalResponse)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)
generation_chain = generation_prompt | llm1.with_structured_output(GenerationResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm1.with_structured_output(UtilityResponse)