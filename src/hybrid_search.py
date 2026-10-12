from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from embed_store import collection, model

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_all_chunks():
    """Fetch all chunks currently stored in the vector database."""
    all_data = collection.get()
    documents = all_data["documents"]
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    return documents, ids, metadatas


def build_bm25_index():
    """Build a BM25 keyword search index from all stored chunks."""
    documents, ids, metadatas = get_all_chunks()
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, documents, ids, metadatas


def bm25_search(query, top_k=5):
    """Keyword-based search using BM25."""
    bm25, documents, ids, metadatas = build_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(documents, scores, metadatas), key=lambda x: x[1], reverse=True)
    top_results = ranked[:top_k]

    return [(doc, meta["source"]) for doc, score, meta in top_results]


def vector_search(query, top_k=5):
    """Semantic search using embeddings."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    documents = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return list(zip(documents, sources))


def hybrid_search(query, top_k=5, rerank_top_n=10):
    """Combine BM25 (keyword) and vector (semantic) search, then rerank."""
    bm25_results = bm25_search(query, top_k=rerank_top_n)
    vector_results = vector_search(query, top_k=rerank_top_n)

    # Combine and deduplicate
    combined = vector_results + bm25_results
    seen = set()
    unique_results = []
    for doc, source in combined:
        if doc not in seen:
            unique_results.append((doc, source))
            seen.add(doc)

    # Rerank using cross-encoder
    pairs = [[query, doc] for doc, source in unique_results]
    scores = reranker.predict(pairs)

    # Sort by reranker score, descending
    reranked = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
    final_results = [item[0] for item in reranked[:top_k]]

    return final_results


if __name__ == "__main__":
    query = "What is entity resolution?"
    results = hybrid_search(query, top_k=5)

    print(f"--- Hybrid + reranked search results for: '{query}' ---\n")
    for i, (doc, source) in enumerate(results, 1):
        print(f"[{i}] (from {source})\n{doc[:200]}...\n")