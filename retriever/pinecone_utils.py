# =========================
# File: retriever/pinecone_utils.py
# =========================

from config import INDEX

def upsert_to_pinecone(docs, embedder):
    vectors = []
    for doc in docs:
        vec = embedder.embed_query(doc.page_content)
        vectors.append({
            "id": f"{doc.metadata['type']}-{doc.metadata['chunk_index']}",
            "values": vec,
            "metadata": {**doc.metadata, "page_content": doc.page_content},  # include content
        })
    INDEX.upsert(vectors=vectors)
