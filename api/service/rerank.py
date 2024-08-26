# from .init import cross_encoder
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def reranking_relevant_documents(query: str, initial_docs : List[Document], rerank_top_k = -1) -> List[Document]:        
    # Prepare pairs for cross-encoder
    pairs = [[query, doc.page_content] for doc in initial_docs]
    
    # Get cross-encoder scores
    scores = cross_encoder.predict(pairs)
    
    # Sort documents by score
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Return top reranked documents
    # return [doc for doc, _ in scored_docs[:rerank_top_k]]
    return [doc for doc, _ in scored_docs]