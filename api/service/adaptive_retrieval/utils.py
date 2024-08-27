
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from config import config as config
from data.pinecone import search as search
from langchain_core.retrievers import BaseRetriever
from service import rerank as rerank  
from model.resource import Resource


"""
Define BaseRetrievalStrategy
"""

class BaseRetrievalStrategy:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)

    def retrieve(self, query, k=4):
        return search.similarity_search(query, k=k)

"""
Define AnalyticalRetrievalStrategy
"""

class RewritingRetriever(BaseRetrievalStrategy):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)
    
    # # Use LLM to enhance the query
    #     enhanced_query_prompt = PromptTemplate(
    #         input_variables=["query"],
    #         template="Enhance this factual query for better information retrieval: {query}"
    #     )
    #     query_chain = enhanced_query_prompt | self.llm
    #     enhanced_query = query_chain.invoke(query).content
    #     print(f'enhande query: {enhanced_query}') 
    
    def rewrite_query(self, original_query):
        """
        Rewrite the original query to improve retrieval.
        
        Args:
        original_query (str): The original user query
        
        Returns:
        str: The rewritten query
        """
        # Create a prompt template for query rewriting
        query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        Rewritten query:"""

        query_rewrite_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=query_rewrite_template
        )

        # Create an LLMChain for query rewriting
        query_rewriter = query_rewrite_prompt | self.llm

        response = query_rewriter.invoke(original_query)
        return response.content

    def retrieve(self, query, k):
        rewritten_query  = self.rewrite_query(query)
        docs = search.similarity_search([rewritten_query], k=k)
        return docs

class StepBackRetriever(BaseRetrievalStrategy):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)
        
    def step_back_prompt(self, query :str):
        # Create a prompt template for step-back prompting
        step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

        Original query: {original_query}

        Step-back query:"""

        step_back_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=step_back_template
        )
        # Create an LLMChain for step-back prompting
        step_back_chain = step_back_prompt | self.llm
        
        def generate_step_back_query(original_query):
            """
            Generate a step-back query to retrieve broader context.
            
            Args:
            original_query (str): The original user query
            
            Returns:
            str: The step-back query
            """
            response = step_back_chain.invoke(original_query)
            return response.content
        
        return generate_step_back_query(query)

    def retrieve(self, query, k):
        step_back_query = self.step_back_prompt(query)
        docs = search.similarity_search([query, step_back_query], k=k)
        return docs

class HyDERetriever(BaseRetrievalStrategy):
    def __init__(self, chunk_size=500):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.8, top_p=0.5)
        self.chunk_size = chunk_size
        
    def generate_hypothetical_document(self, query):
        hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
        hyde_chain = hyde_prompt | self.llm
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        docs = search.similarity_search([query, hypothetical_doc], k=k)
        return docs
    
    
"""
Define AnalyticalRetrievalStrategy
"""

class multiple_queries(BaseModel):  
    # setup: str = Field(description="Original query")
    query1: str  = Field(description="query 1")
    query2: str  = Field(description="query 2")
    query3: str  = Field(description="query 3")

class FusionRetriever(BaseRetrievalStrategy):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)
        
        
    def get_generated_queries(self, query, k_queries = 3):
        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        structured_llm = self.llm.with_structured_output(multiple_queries)
        generate_queries = (
            prompt_rag_fusion 
            | structured_llm
        )
        result = generate_queries.invoke({"question" : query})
        return [result.query1,result.query2,result.query3]
    
    def retrieve(self, query, k=4):
        queries = self.get_generated_queries(query)
        print("Generated_queries : ", queries)
        docs = search.similarity_search(queries, k=k)
        return docs
    
class SubQueryDecompositionRetriever(BaseRetrievalStrategy):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)
        
    def get_generated_queries(self, query, k_queries = 3):
        # Create a prompt template for sub-query decomposition
        subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 3 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
        
        Original query: {original_query}

        example: What are the impacts of climate change on the environment?

        Sub-queries:
        1. What are the impacts of climate change on biodiversity?
        2. How does climate change affect the oceans?
        3. What are the effects of climate change on agriculture?
        """
        
        subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 3 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
        
        Original query: {original_query}
        """

        #4. What are the impacts of climate change on human health?
        subquery_decomposition_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=subquery_decomposition_template
        )
        structured_llm = self.llm.with_structured_output(multiple_queries)
        # Create an LLMChain for sub-query decomposition
        subquery_decomposer_chain = (subquery_decomposition_prompt | structured_llm)
        result = subquery_decomposer_chain.invoke({"original_query" : query})
        return [result.query1,result.query2,result.query3]
    
    def retrieve(self, query, k=4):
        queries = self.get_generated_queries(query)
        print("Generated sub-queries : ", queries)
        docs = search.similarity_search(queries, k=k)
        return docs