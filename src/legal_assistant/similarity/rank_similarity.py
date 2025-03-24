from sentence_transformers import SentenceTransformer, util

# Load the lightweight embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def rank_by_similarity(query: str, documents: list[str]) -> list[tuple]:
    # Encode query and documents
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    doc_embeddings = embedder.encode(documents, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    # Pair documents with scores and sort (convert similarities to a list of floats)
    scored_docs = sorted(zip(documents, similarities.tolist()), key=lambda x: x[1], reverse=True)
    return scored_docs

def get_cosine_similarity_score(query: str, document: str) -> float:
    # Encode query and documents
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    doc_embeddings = embedder.encode(document, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    return similarity.item()

if __name__ == "__main__":
    # Example query and documents
    query = "What are the legal requirements for filing a medical malpractice lawsuit in California?"
    documents = [
        # Partially relevant (mentions malpractice but not California)
        "The National Medical Malpractice Reform Act outlines federal guidelines for expert witness qualifications in malpractice cases, requiring board certification in the relevant specialty.",
        
        # Relevant jurisdiction but different legal area
        "California Penal Code § 803 specifies statute of limitations for felony assault cases, requiring filing within 3 years of incident.",
        
        # Contains keywords but irrelevant context
        "To file a malpractice complaint with the California Medical Board, use Form MB-20. This applies to licensing issues, not lawsuits.",
        
        # Similar keywords but different domain
        "California real estate malpractice suits require proving breach of fiduciary duty under Business & Professions Code § 10176.",
        
        # Indirectly related (procedure without specifics)
        "All California lawsuits require filing fees, a properly formatted complaint, and service of process to defendants.",
        
        # Red herring (contains 'medical' and 'California')
        "California Health & Safety Code § 1234 regulates medical waste disposal by healthcare facilities.",
        
        # Directly relevant (should rank highest)
        "California Code of Civil Procedure § 340.5: Medical malpractice actions shall be commenced within three years after the date of injury or one year after the plaintiff discovers the injury, whichever occurs first. Requires certificate of merit from a medical expert.",
    ]
    
    # Rank documents by similarity
    ranked_results = rank_by_similarity(query, documents)
    
    # Print the results
    print("Query:", query)
    print("\nRanked Documents:")
    for doc, score in ranked_results:
        # Use score.item() if score is still a tensor; here we assume it's a float
        print(f"Score: {score:.4f}, Document: {doc}")