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



class categories_options(BaseModel):
        category: str = Field(description="The category of the query, the options are: Factual, Analytical", example="Factual")


class QueryClassifier:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8, top_p=0.5)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical.\nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)


    def classify(self, query):
        print("clasiffying query")
        return self.chain.invoke(query).category

"""
Define BaseRetrievalStrategy
"""

class BaseRetrievalStrategy:
    def __init__(self):
        self.search_engine = search
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)


    def retrieve(self, query, k=4):
        return self.search_engine.similarity_search(query, k=k)

    
class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving factual")
        # Use LLM to enhance the query
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'enhande query: {enhanced_query}')

        # Retrieve documents using the enhanced query
        docs = self.search_engine.similarity_search([enhanced_query], k=k)
        return [doc for doc in docs]


"""
Define AnalyticalRetrievalStrategy
"""

class multiple_queries(BaseModel):  
    # setup: str = Field(description="Original query")
    query1: str  = Field(description="query 1")
    query2: str  = Field(description="query 2")
    query3: str  = Field(description="query 3")

def get_generated_queries(query, k_queries = 3):
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8, top_p=0.5)
    structured_llm = llm.with_structured_output(multiple_queries)
    generate_queries = (
        prompt_rag_fusion 
        | structured_llm
    )
    result = generate_queries.invoke({"question" : query})
    return [result.query1,result.query2,result.query3]
    
class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        queries = get_generated_queries(query)
        print("Generated_queries : ", queries)
        docs = self.search_engine.similarity_search(queries, k=k)
        return docs
    


   
"""
AdaptiveRetriever
"""

class AdaptiveRetriever:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(),
            "Analytical": AnalyticalRetrievalStrategy(),
            # "Opinion": OpinionRetrievalStrategy(texts),
            # "Contextual": ContextualRetrievalStrategy(texts)
        }

    def get_relevant_documents(self, query: str, k:int = 3, mode: str = "Auto") -> List[Document]:
        if mode == "Auto" or mode not in ['Factual', 'Analytical', 'Auto']:
            category = self.classifier.classify(query)
        else:
            category = mode
        print("Using : ", category)
        strategy = self.strategies[category]
        return strategy.retrieve(query, k)
    
# Define aditional retriever that inherits from langchain BaseRetriever
class PydanticAdaptiveRetriever():
    def __init__(self, adaptive_retriever):
        self.adaptive_retriever: AdaptiveRetriever = adaptive_retriever

    def get_relevant_documents(self, query: str, k:int = 3, mode: str = "Auto") -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query, k, mode)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
    
    
# Define the Adaptive RAG class  

def results_to_model(result:Document) -> Resource:
    return Resource(
                topic  = result.metadata["topic"],
                title = result.metadata["title"],
                principle   = result.metadata["principle"]
            )

class AdaptiveRAG:
    def __init__(self):
        adaptive_retriever = AdaptiveRetriever()
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8, top_p=0.5)
        
        # Create a custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create the LLM chain
        self.llm_chain = prompt | self.llm | StrOutputParser()
        
    def answer(self, query: str, k:int = 3, rerank_mode : bool = True, query_category: str = "Auto") -> str:
        docs = self.retriever.get_relevant_documents(query, k, query_category)
        print("Num docs : ", len(docs))
        if rerank_mode:
            docs = rerank.reranking_relevant_documents(query, docs)
        resources = [results_to_model(doc) for doc in docs]
        # print(docs)
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data), resources