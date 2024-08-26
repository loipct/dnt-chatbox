from .chains import retrieval_chain, relevance_chain,\
        generation_chain, support_chain, utility_chain
        
from data.pinecone import search as search
import time
       
def self_rag(query, top_k):
    print(f"\nProcessing query: {query}")
    
    # Step 1: Determine if retrieval is necessary
    print("Step 1: Determining if retrieval is necessary...")
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()
    print(f"Retrieval decision: {retrieval_decision}")
    
    if retrieval_decision == 'yes':
        # Step 2: Retrieve relevant documents
        print("Step 2: Retrieving relevant documents...")
        docs = search.similarity_search([query], k = top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"Retrieved {len(contexts)} documents")
        
        # Step 3: Evaluate relevance of retrieved documents
        print("Step 3: Evaluating relevance of retrieved documents...")
        relevant_contexts = []
        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance = relevance_chain.invoke(input_data).response.strip().lower()
            print(f"Document {i+1} relevance: {relevance}")
            if relevance == 'relevant':
                relevant_contexts.append(context)
            time.sleep(4)
        
        print(f"Number of relevant contexts: {len(relevant_contexts)}")
        
        # If no relevant contexts found, generate without retrieval
        if not relevant_contexts:
            print("No relevant contexts found. Generating without retrieval...")
            input_data = {"query": query, "context": "No relevant context found."}
            return generation_chain.invoke(input_data).response
        time.sleep(4)
        # Step 4: Generate response using relevant contexts
        print("Step 4: Generating responses using relevant contexts...")
        responses = []
        for i, context in enumerate(relevant_contexts):
            print(f"Generating response for context {i+1}...")
            input_data = {"query": query, "context": context}
            response = generation_chain.invoke(input_data).response
            time.sleep(4)
            
            # Step 5: Assess support
            print(f"Step 5: Assessing support for response {i+1}...")
            input_data = {"response": response, "context": context}
            support = support_chain.invoke(input_data).response.strip().lower()
            time.sleep(4)
            print(f"Support assessment: {support}")
            
            # Step 6: Evaluate utility
            print(f"Step 6: Evaluating utility for response {i+1}...")
            input_data = {"query": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response)
            print(f"Utility score: {utility}")
            
            responses.append((response, support, utility))
        
        # Select the best response based on support and utility
        print("Selecting the best response...")
        best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
        print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
        return best_response[0]
    else:
        # Generate without retrieval
        print("Generating without retrieval...")
        input_data = {"query": query, "context": "No retrieval necessary."}
        return generation_chain.invoke(input_data).response