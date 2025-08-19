# =========================
# File: processing/chunking.py
# =========================

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents([text])
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["type"] = "text"
    return docs


def chunk_images(descriptions):
    image_docs = []
    for i, desc in enumerate(descriptions):
        metadata = {
            "chunk_index": i,
            "type": "image",
            "page": desc["page"],
            "figure_number": desc["figure_number"],
        }
        image_docs.append(Document(page_content=desc["description"], metadata=metadata))
    return image_docs