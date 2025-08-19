# =========================
# File: retriever/embeddings.py
# =========================

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def get_embeddings():
    # Initialize and return the embeddings model
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large") # change if needed