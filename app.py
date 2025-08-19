# =========================
# File: app.py
# =========================

import streamlit as st
from processing.pipeline import process_pdf
from ui.streamlit_ui import run_streamlit_app


def main():
    st.title("📤 Upload a PDF to Start")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"]) 

    if uploaded_file:
        if "pdf_data" not in st.session_state:
            file_bytes = uploaded_file.read()
            st.session_state.pdf_data = process_pdf(file_bytes)
        run_streamlit_app(st.session_state.pdf_data, uploaded_file.name)
    else:
        st.info("👆 Please upload a PDF.")


if __name__ == "__main__":
    main()
















# =========================
# File: processing/__init__.py
# =========================
# (empty)



# =========================
# File: retriever/__init__.py
# =========================
# (empty)



# =========================
# File: qa/__init__.py
# =========================
# (empty)




# =========================
# File: ui/__init__.py
# =========================
# (empty)



# # =========================
# # File: requirements.txt
# # =========================
# streamlit==1.36.0
# pytesseract==0.3.10
# pdf2image==1.17.0
# Pillow==10.4.0
# pymupdf==1.24.9
# langchain==0.2.12
# langchain-google-genai==1.0.6
# google-generativeai==0.7.2
# pinecone-client==5.0.1
# pydantic==2.8.2
# sentence-transformers==3.0.1
# # torch is required by sentence-transformers; install per your platform
# # pip install torch --index-url https://download.pytorch.org/whl/cu121  (for CUDA 12.1) or plain `pip install torch`

# # =========================
# # File: README.md
# # =========================
# # Multimodal Agentic RAG for PDF Q&A (Streamlit)

# ## Quickstart
# ```bash
# python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install -r requirements.txt
# # Ensure Tesseract is installed at C:\\Program Files\\Tesseract-OCR\\tesseract.exe (as hardcoded)
# streamlit run app.py
# ```

# ## Notes
# - Poppler is required for `pdf2image` on some systems.
# - All API keys and model names are centralized in `config.py` **exactly as in your original code**.
# - This repo mirrors your monolith logic with minimal structural changes: only modularization.
