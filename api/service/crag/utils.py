import os
import sys
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Tuple
from langchain_community.tools import DuckDuckGoSearchResults
from data.pinecone import search as search


llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)
llm1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)
#Define retrieval evaluator, knowledge refinement and query rewriter llm chains
# Retrieval Evaluator
class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of the document to the query. the score should be between 0 and 1.")

def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate(
        input_variables=["query", "document"],
        template="On a scale from 0 to 1, how relevant is the following document to the query? Query: {query}\nDocument: {document}\nRelevance score:"
    )
    chain = prompt | llm1.with_structured_output(RetrievalEvaluatorInput)
    input_variables = {"query": query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result

# Knowledge Refinement
class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The document to extract key information from.")
def knowledge_refinement(document: str) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["document"],
        template="Extract the key information from the following document in bullet points:\n{document}\nKey points:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    input_variables = {"document": document}
    result = chain.invoke(input_variables).key_points
    return [point.strip() for point in result.split('\n') if point.strip()]

# Web Search Query Rewriter
class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")
def rewrite_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query": query}
    return chain.invoke(input_variables).query.strip()


def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """
    Parse a JSON string of search results into a list of title-link tuples.

    Args:
        results_string (str): A JSON-formatted string containing search results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.
                               If parsing fails, an empty list is returned.
    """
    try:
        # Attempt to parse the JSON string
        results = json.loads(results_string)
        # Extract and return the title and link from each result
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        # Handle JSON decoding errors by returning an empty list
        print("Error parsing search results. Returning empty list.")
        return []
    
def retrieve_documents(query: str, k: int = 3) -> List[str]:
    """
    Retrieve documents based on a query using a FAISS index.

    Args:
        query (str): The query string to search for.
        faiss_index (FAISS): The FAISS index used for similarity search.
        k (int): The number of top documents to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of the retrieved document contents.
    """
    docs = search.similarity_search([query], k = k)
    return [doc.page_content for doc in docs]

def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    Evaluate the relevance of documents based on a query.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """
    return [retrieval_evaluator(query, doc) for doc in documents]

def perform_web_search(query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Perform a web search based on a query.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: 
            - A list of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources

def generate_response(query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
    """
    Generate a response to a query using knowledge and sources.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template="Based on the following knowledge, answer the query. Include the sources with their links (if available) at the end of your answer:\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content