# Bangla/English PDF RAG Pipeline with Streamlit PDF Upload
# Uses OCR to convert the pages into images and for text extraction from those images in the PDF
# Uses VLM to generate Image descriptions
# Includes: Extraction, Cleaning, Chunking, Embedding, Vector Store, QA (Langchain + Streamlit)
# Entire PDF preprocessing and chunking (without metadata)

import os
import re
import io
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_bytes
from google.generativeai import configure, GenerativeModel
from langchain.docstore.document import Document

# Configure Google API for Gemini Vision
configure(api_key="")
vision_model = GenerativeModel("gemini-2.5-pro")  # Vision-capable model

# === 1. EXTRACT TEXT + OCR FULL PAGES + IMAGES ONLY === #
def extract_and_clean_text(file_bytes):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ---------- OCR Full Pages ----------
    pages_as_images = convert_from_bytes(file_bytes, dpi=300)
    full_text = ""
    for page_img in pages_as_images:
        ocr_result = pytesseract.image_to_string(page_img, lang="ben+eng")
        full_text += ocr_result + "\n"

    # ---------- Extract images ----------
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    extracted_images = []

    for page_index, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                extracted_images.append({
                    "page": page_index,
                    "figure_number": None,  # Will detect from OCR
                    "image": pil_img
                })
            except Exception as e:
                st.warning(f"⚠️ Image extraction failed on page {page_index}: {e}")

    # ---------- Detect Figure Numbers in OCR text ----------
    #figure_pattern = re.compile(r"(Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    figure_pattern = re.compile(r"(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    figure_matches = figure_pattern.findall(full_text)
    fig_counter = 0
    for img_data in extracted_images:
        if fig_counter < len(figure_matches):
            img_data["figure_number"] = figure_matches[fig_counter]
            fig_counter += 1

    # ---------- Clean OCR text ----------
    text = re.sub(r"\n\s*\d+\s*\n", "\n", full_text)  # Remove isolated page numbers
    text = re.sub(r"\n+", "\n", text)  # Remove excessive newlines

    return text.strip(), extracted_images


# === 2. IMAGE DESCRIPTION USING GEMINI VISION === #
def describe_images_with_llm(extracted_images):
    image_descriptions = []
    for img_data in extracted_images:
        try:
            img_buffer = io.BytesIO()
            img_data["image"].save(img_buffer, format="PNG")
            img_buffer.seek(0)

            prompt = (
                "Provide short description of this figure."
                "Include any visible text, diagrams, charts, and their meaning."
            )

            response = vision_model.generate_content(
                [prompt, {"mime_type": "image/png", "data": img_buffer.read()}]
            )

            description = response.text.strip() if response.text else "No description generated."
            image_descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data["figure_number"],
                "description": description
            })

        except Exception as e:
            image_descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data["figure_number"],
                "description": f"Error describing image: {e}"
            })

    return image_descriptions


# === 3. CHUNKING WITH METADATA === #
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["type"] = "text"
    return docs


def chunk_images(image_descriptions):
    image_docs = []
    for i, desc in enumerate(image_descriptions):
        metadata = {
            "chunk_index": i,
            "type": "image",
            "page": desc["page"],
            "figure_number": desc["figure_number"]
        }
        image_docs.append(Document(page_content=desc["description"], metadata=metadata))
    return image_docs


@st.cache_resource(show_spinner="🔍 Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


@st.cache_resource(show_spinner="📦 Building vector database...")
def build_vector_db(_docs, _embeddings):
    return FAISS.from_documents(_docs, _embeddings)


@st.cache_resource(show_spinner="🤖 Setting up Gemini QA chain...")
def build_qa_chain(_vector_db):
    return RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="models/gemini-2.5-pro",
            google_api_key=""
        ),
        retriever=_vector_db.as_retriever(),
        return_source_documents=True
    )


@st.cache_resource(show_spinner="📄 Processing uploaded PDF...")  
def process_pdf(file_bytes):
    text, images = extract_and_clean_text(file_bytes)
    image_descriptions = describe_images_with_llm(images)
    text_chunks = chunk_text(text)
    image_chunks = chunk_images(image_descriptions)
    all_chunks = text_chunks + image_chunks
    embedder = get_embeddings()
    vector_db = build_vector_db(all_chunks, embedder)
    qa_chain = build_qa_chain(vector_db)
    return {
        "text": text,
        "images": images,
        "image_descriptions": image_descriptions,
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "all_chunks": all_chunks,
        "embedder": embedder,
        "vector_db": vector_db,
        "qa_chain": qa_chain
    }


# === 7. STREAMLIT INTERFACE === #
def run_streamlit_app(pdf_data, pdf_name):
    st.title("📘 PDF Question Answering (RAG) + Image Understanding")
    st.markdown(f"#### {pdf_name}")

    # Show preprocessed text chunks (now includes image chunks)
    with st.expander("🔎 Show Preprocessed Text Chunks"):
       #if st.button("📂 View All Text Chunks"):
            for i, chunk in enumerate(pdf_data["all_chunks"]):  # ✅ Includes image chunks now
                label = f"**{chunk.metadata.get('type').capitalize()} Chunk {i}:**"
                if chunk.metadata.get("type") == "image":
                    label += f" (Page {chunk.metadata.get('page')}, Fig: {chunk.metadata.get('figure_number') or 'N/A'})"
                st.markdown(label)
                st.write(chunk.page_content)
                st.markdown("---")

    # Show extracted images with metadata
    with st.expander("🖼 View Extracted Images"):
        if pdf_data["images"]:
            for i, img_data in enumerate(pdf_data["images"], start=1):
                caption = f"Image {i} | Page {img_data['page']}"
                if img_data["figure_number"]:
                    caption += f" | {img_data['figure_number']}"
                st.image(img_data["image"], caption=caption, use_container_width=True)
        else:
            st.write("No images found in PDF.")

    # Show generated image descriptions from LLM
    with st.expander("📝 Image Descriptions from LLM"):
        for i, desc in enumerate(pdf_data["image_descriptions"], start=1):
            label = f"Image {i} | Page {desc['page']}"
            if desc["figure_number"]:
                label += f" | {desc['figure_number']}"
            st.markdown(f"**{label}**")
            st.write(desc["description"])
            st.markdown("---")

    # Ask a question
    question = st.text_input("(Type your text question in English/Bangla)")

    if question:
        if "last_answer" not in st.session_state or st.session_state.last_question != question:
            with st.spinner("🤔 Generating answer..."):
                result = pdf_data["qa_chain"]({"query": question})
                st.session_state.last_answer = result
                st.session_state.last_question = question
        else:
            result = st.session_state.last_answer  # ✅ Avoid regenerating

        st.success(f"📝 উত্তর: {result['result']}")

        if st.button("📌 Show Source Chunks & Images Used"):
            source_chunks = result.get("source_documents", [])
            text_ids = []
            image_info = []
            for doc in source_chunks:
                if doc.metadata.get("type") == "text":
                    text_ids.append(doc.metadata.get("chunk_index"))
                elif doc.metadata.get("type") == "image":
                    image_info.append((
                        doc.metadata.get("page"),
                        doc.metadata.get("figure_number"),
                        doc.metadata.get("chunk_index")
                    ))
            st.markdown("**Text Chunks Used:** " + ", ".join(map(str, text_ids)) if text_ids else "None")
            if image_info:
                st.markdown("**Images Used:**")
                for idx, (page, fig, chunk_idx) in enumerate(image_info, start=1):
                    st.markdown(f"- Image {idx} | Page {page} | {fig or 'No Figure Label'}")


# === 8. MAIN EXECUTION === #
def main():
    st.title("📤 Upload a PDF to Start")

    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

    if uploaded_file:
        if "pdf_data" not in st.session_state:
            #with st.spinner("📄 Processing uploaded PDF..."):
                file_bytes = uploaded_file.read()
                st.session_state.pdf_data = process_pdf(file_bytes)

        run_streamlit_app(st.session_state.pdf_data, pdf_name=uploaded_file.name)
    else:
        st.info("👆 Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()

