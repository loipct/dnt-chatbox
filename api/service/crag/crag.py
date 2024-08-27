from .utils import *

def crag_process(query: str, k:int = 5) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.

    Args:
        query (str): The query string to process.

    Returns:
        str: The generated response based on the query.
    """
    print(f"\nProcessing query: {query}")

    # Retrieve and evaluate documents
    retrieved_docs = retrieve_documents(query, k)
    eval_scores = evaluate_documents(query, retrieved_docs)
    
    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print(f"Evaluation scores: {eval_scores}")

    # Determine action based on evaluation scores
    max_score = max(eval_scores)
    sources = []
    
    if max_score > 0.7:
        print("\nAction: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        print("\nAction: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        print("\nAction: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        # Refine the retrieved knowledge
        retrieved_knowledge = knowledge_refinement(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    print("\nFinal knowledge:")
    print(final_knowledge)
    
    print("\nSources:")
    for title, link in sources:
        print(f"{title}: {link}" if link else title)

    # Generate response
    print("\nGenerating response...")
    response = generate_response(query, final_knowledge, sources)

    print("\nResponse generated")
    return response