# =========================
# File: processing/pipeline.py
# =========================

import streamlit as st
from processing.ocr import extract_and_clean_text
from processing.images import describe_images_with_llm
from processing.chunking import chunk_text, chunk_images
from retriever.embeddings import get_embeddings
from retriever.pinecone_utils import upsert_to_pinecone
from qa.chain import build_qa_chain


@st.cache_resource
def process_pdf(file_bytes):
    text, images = extract_and_clean_text(file_bytes)
    img_desc = describe_images_with_llm(images)
    #img_desc = [] # stop generating image description
    text_chunks = chunk_text(text)
    img_chunks = chunk_images(img_desc)
    all_chunks = text_chunks + img_chunks

    embedder = get_embeddings()
    upsert_to_pinecone(all_chunks, embedder)
    qa_chain = build_qa_chain(embedder)

    return {
        "text": text,
        "images": images,
        "image_descriptions": img_desc,
        "text_chunks": text_chunks,
        "image_chunks": img_chunks,
        "all_chunks": all_chunks,
        "embedder": embedder,
        "qa_chain": qa_chain,
    }