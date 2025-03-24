from sentence_transformers import CrossEncoder

# Add this to your code
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_docs(query: str, top_docs: list[str], top_k=3) -> list[str]:
    pairs = [(query, doc) for doc in top_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(top_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# Updated pipeline
def select_top_docs(query: str, raw_documents: list[str]) -> list[str]:
    # Step 1: Get top 10 candidates via embeddings
    scored_docs = rank_by_similarity(query, raw_documents)
    top_candidates = [doc for doc, _ in scored_docs[:10]]
    
    # Step 2: Re-rank with cross-encoder
    reranked = rerank_docs(query, top_candidates)
    return reranked[:3]