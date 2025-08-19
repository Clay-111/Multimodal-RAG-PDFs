# =========================
# File: qa/chain.py
# =========================

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import INDEX, GOOGLE_API_KEY_MAIN, GEMINI_TEXT_MODEL_NAME
from retriever.pinecone_retriever import PineconeRetriever


@st.cache_resource
def build_qa_chain(_embedder):
    retriever = PineconeRetriever(INDEX, _embedder)
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model=GEMINI_TEXT_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY_MAIN,
        ),
        retriever=retriever,
        return_source_documents=True,
    )
    return qa