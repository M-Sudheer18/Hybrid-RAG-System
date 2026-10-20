
from config.settings import settings
from src.retrieval.retriever import Retriever
from src.generation.reranker import CrossEncoderReranker

query = "What do the QualiRpar, QualiCert, and RecQ labels verify?"

# Stage 1: Hybrid retrieval
retriever = Retriever()

candidates = retriever.retrieve(
    query=query,
    dense_top_k=settings.TOP_K_SEARCH,
    sparse_top_k=settings.TOP_K_SPARSE,
    final_top_n=15,
)

print(f"\nHybrid candidates: {len(candidates)}")

# Stage 2: Cross-encoder reranking
reranker = CrossEncoderReranker.get_instance()

top_chunks = reranker.rerank(
    query=query,
    chunks=candidates,
    top_n=settings.TOP_N_RERANK,
)

print(f"Chunks passed forward: {len(top_chunks)}")

# Inspect the exact chunks selected for answer generation
for rank, chunk in enumerate(top_chunks, start=1):
    metadata = chunk.get("metadata", {})

    filename = (
        metadata.get("filename")
        or metadata.get("file_name")
        or metadata.get("source")
        or "Unknown"
    )

    print("\n" + "=" * 70)
    print(f"RERANKED RESULT {rank}")
    print(f"File: {filename}")
    print(f"Page: {metadata.get('page', metadata.get('page_number', 'Unknown'))}")
    print(f"Score: {chunk.get('cross_encoder_score', 'N/A')}")
    print(f"Content:\n{chunk.get('content', '')[:1200]}")
