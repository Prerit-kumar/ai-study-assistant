import os
import threading

from rank_bm25 import BM25Okapi

from embed_store import collection, model

# LIGHT_MODELS=1 (on Render) runs the same reranker on ONNX, without PyTorch, to fit in 512 MB.
if os.getenv("LIGHT_MODELS") == "1":
    from light_models import Reranker
    reranker = Reranker()
else:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# The BM25 index is built once and rebuilt only when the number of chunks changes (e.g. after an upload).
_bm25_cache = {"count": None, "index": None}
_bm25_lock = threading.Lock()


def get_all_chunks():
    """Fetch all chunks currently stored in the vector database."""
    all_data = collection.get()
    documents = all_data["documents"]
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    return documents, ids, metadatas


def build_bm25_index():
    """BM25 keyword index over all stored chunks, cached until the database changes."""
    with _bm25_lock:
        count = collection.count()
        if _bm25_cache["count"] != count:
            documents, ids, metadatas = get_all_chunks()
            tokenized_docs = [doc.lower().split() for doc in documents]
            _bm25_cache["index"] = (BM25Okapi(tokenized_docs), documents, ids, metadatas)
            _bm25_cache["count"] = count
        return _bm25_cache["index"]


def bm25_search(query, top_k=5, sources=None):
    """Keyword-based search using BM25. `sources`: only search these documents (None = all)."""
    bm25, documents, ids, metadatas = build_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(documents, scores, metadatas), key=lambda x: x[1], reverse=True)
    if sources:
        allowed = set(sources)
        ranked = [r for r in ranked if r[2]["source"] in allowed]
    top_results = ranked[:top_k]

    return [(doc, meta["source"]) for doc, score, meta in top_results]


def vector_search(query, top_k=5, sources=None):
    """Semantic search using embeddings. `sources`: only search these documents (None = all)."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"source": {"$in": list(sources)}} if sources else None,
    )
    documents = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return list(zip(documents, sources))


def hybrid_search(query, top_k=5, rerank_top_n=10, sources=None):
    """Combine BM25 (keyword) and vector (semantic) search, then rerank.
    `sources`: only search these documents (None or empty = all documents)."""
    bm25_results = bm25_search(query, top_k=rerank_top_n, sources=sources)
    vector_results = vector_search(query, top_k=rerank_top_n, sources=sources)

    # Combine and deduplicate
    combined = vector_results + bm25_results
    seen = set()
    unique_results = []
    for doc, source in combined:
        if doc not in seen:
            unique_results.append((doc, source))
            seen.add(doc)

    if not unique_results:
        return []

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