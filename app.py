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

