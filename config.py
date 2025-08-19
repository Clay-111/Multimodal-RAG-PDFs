# =========================
# File: config.py
# =========================

import os
from google.generativeai import configure, GenerativeModel
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# ⚠️ API KEYS & MODELS 
# -------------------------
GOOGLE_API_KEY_MAIN = "AIzaSyDaWk_DZqXwQDCBX4iaddu-250lRmDK0Ho"  # LLM API used for configure() and QA chain
GOOGLE_API_KEY_SECOND = "AIzaSyDaWk_DZqXwQDCBX4iaddu-250lRmDK0Ho"  # VLM API used in multi_image_query
GEMINI_VLM_MODEL_NAME = "gemini-2.5-pro"                            # VLM via VLM API 
GEMINI_TEXT_MODEL_NAME = "models/gemini-2.5-pro"  # LLM via LLM API
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # Embedding model

# Configure Google Generative AI
configure(api_key=GOOGLE_API_KEY_MAIN)
VISION_MODEL = GenerativeModel(GEMINI_VLM_MODEL_NAME)

# PineconeDB v2 configuration
PINECONE_API_KEY = "pcsk_4FNN28_3BsvDSQeCVP3SeCbkBNE78Y3Q4aX4D5RFqm7iaL4hGabRGvb1FQzDaE38HH5dAb" # change as needed
INDEX_NAME = "rag" # change as needed

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