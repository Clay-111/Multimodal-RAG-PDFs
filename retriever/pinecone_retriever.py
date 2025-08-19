# =========================
# File: retriever/pinecone_retriever.py
# =========================

from pydantic import PrivateAttr
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document


class PineconeRetriever(BaseRetriever):
    _index: any = PrivateAttr()
    _embedder: any = PrivateAttr()
    _top_k: int = PrivateAttr()

    def __init__(self, index, embedder, top_k=5):
        super().__init__()
        self._index = index
        self._embedder = embedder
        self._top_k = top_k

    def get_relevant_documents(self, query: str):
        q_vec = self._embedder.embed_query(query)
        results = self._index.query(vector=q_vec, top_k=self._top_k, include_metadata=True)
        docs = []
        for match in results.matches:
            content = match.metadata.get("page_content", "")  # <-- use stored content
            docs.append(Document(page_content=content, metadata=match.metadata))
        return docs