# =========================
# File: config.py
# =========================

import os
from google.generativeai import configure, GenerativeModel
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# ⚠️ API KEYS & MODELS 
# -------------------------
GOOGLE_API_KEY_MAIN = "YOUR_GEMINI_LLM_API_KEY"  # LLM API used for configure() and QA chain
GOOGLE_API_KEY_SECOND = "YOUR_GEMINI_VLM_API_KEY"  # VLM API used in multi_image_query
GEMINI_VLM_MODEL_NAME = "YOUR_GEMINI_VLM_MODEL_NAME"  # VLM via VLM API || Example - (gemini-2.5-pro)
GEMINI_TEXT_MODEL_NAME = "YOUR_GEMINI_LLM_MODEL_NAME"  # LLM via LLM API || Example - (gemini-2.5-pro)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # Embedding model

# Configure Google Generative AI
configure(api_key=GOOGLE_API_KEY_MAIN)
VISION_MODEL = GenerativeModel(GEMINI_VLM_MODEL_NAME)

# PineconeDB v2 configuration
PINECONE_API_KEY = "YOUR_PineConeDB_API" # change as needed
INDEX_NAME = "YOUR_PineConeDB_index_name" # change as needed

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it doesn't exist (dimension 1024 for multilingual-e5-large) check retrieval/embeddings.py
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
INDEX = pc.Index(INDEX_NAME)
