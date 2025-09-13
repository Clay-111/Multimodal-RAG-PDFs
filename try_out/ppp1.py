# Bangla/English PDF RAG Pipeline with Streamlit PDF Upload
# Uses OCR to convert the pages into images and for text extraction from those images in the PDF
# Uses VLM to generate Image descriptions.
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

# Configure Google API for Gemini Vision Model
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
                    "figure_number": None,
                    "image": pil_img
                })
            except Exception as e:
                st.warning(f"⚠️ Image extraction failed on page {page_index}: {e}")

    # ---------- Detect Figure Numbers in OCR text ----------
    figure_pattern = re.compile(r"(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    figure_matches = figure_pattern.findall(full_text)
    fig_counter = 0
    for img_data in extracted_images:
        if fig_counter < len(figure_matches):
            img_data["figure_number"] = figure_matches[fig_counter]
            fig_counter += 1

    # ---------- Clean OCR text ----------
    text = re.sub(r"\n\s*\d+\s*\n", "\n", full_text)
    text = re.sub(r"\n+", "\n", text)

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
    #image_descriptions = describe_images_with_llm(images)
    image_descriptions = [] 
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


# === HELPER FUNCTION TO GET IMAGE DESCRIPTIONS FOR MULTIPLE IMAGES ON A PAGE ===
def get_image_descriptions_multi(pdf_data, query):
    # Count images per page for numbering
    page_image_counts = {}
    for img in pdf_data["images"]:
        page = img["page"]
        page_image_counts[page] = page_image_counts.get(page, 0) + 1

    # Compose info string for prompt
    page_img_info = []
    for page in sorted(page_image_counts.keys()):
        page_img_info.append(f"Page {page} has {page_image_counts[page]} images.")
    page_img_info_str = "\n".join(page_img_info)

    # LLM prompt to extract page and possibly multiple image numbers from user query
    prompt = f"""
You are given a user query about a PDF document.

Available image info:
{page_img_info_str}

User query:
\"\"\"{query}\"\"\"

Does the user ask specifically about one or more images on a page?
If yes, reply exactly in this JSON format:
{{"page": <page_number>, "images": [<image_number1>, <image_number2>, ...]}}
If no specific image is mentioned, reply with:
No specific image
"""

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro",
                                 google_api_key="")
    try:
        response = llm.predict(prompt)
        text = response.strip()

        if text.lower().startswith("no specific image"):
            return None

        # Try to parse JSON from response
        import json
        try:
            parsed = json.loads(text)
            page_num = parsed.get("page")
            images_list = parsed.get("images", [])
            if (not isinstance(page_num, int)) or (not isinstance(images_list, list)):
                return None

            # Validate image numbers
            imgs_on_page = [img for img in pdf_data["images"] if img["page"] == page_num]
            descriptions = []
            for img_num in images_list:
                if isinstance(img_num, int) and 0 < img_num <= len(imgs_on_page):
                    # Find global index of this image on this page
                    count = 0
                    global_index = None
                    for idx, img in enumerate(pdf_data["images"]):
                        if img["page"] == page_num:
                            count += 1
                            if count == img_num:
                                global_index = idx
                                break
                    if global_index is not None and global_index < len(pdf_data["image_descriptions"]):
                        descriptions.append(f"Image {img_num} on Page {page_num}:\n{pdf_data['image_descriptions'][global_index]['description']}")
            if descriptions:
                return "\n\n---\n\n".join(descriptions)
            else:
                return f"No descriptions found for the requested images on page {page_num}."
        except Exception:
            return None
    except Exception:
        return None


# === 7. STREAMLIT INTERFACE === #
def run_streamlit_app(pdf_data, pdf_name):
    st.title("📘 Multimodal Agentic RAG for PDF Q&A")
    st.markdown(f"#### {pdf_name}")

    # Show preprocessed text chunks (now includes image chunks)
    with st.expander("🔎 Show Preprocessed Text Chunks"):
        for i, chunk in enumerate(pdf_data["all_chunks"]):
            label = f"**{chunk.metadata.get('type').capitalize()} Chunk {i}:**"
            if chunk.metadata.get("type") == "image":
                label += f" (Page {chunk.metadata.get('page')}, Fig: {chunk.metadata.get('figure_number') or 'N/A'})"
            st.markdown(label)
            st.write(chunk.page_content)
            st.markdown("---")

    # Show extracted images WITH descriptions in one section
    with st.expander("🖼 View Extracted Images and Their Descriptions"):
        if pdf_data["images"]:
            page_image_count = {}
            for img_data in pdf_data["images"]:
                page = img_data["page"]
                page_image_count[page] = page_image_count.get(page, 0) + 1
                image_num = page_image_count[page]

                st.image(img_data["image"], caption=f"Page {page} | Image {image_num}", use_container_width=True)

                # Description inside collapsible expander
                with st.expander(f"Show Description for Image {image_num} on Page {page}"):
                    matching_desc = next(
                        (desc["description"] for desc in pdf_data["image_descriptions"] if desc["page"] == img_data["page"]),
                        "No description available."
                    )
                    st.markdown(matching_desc)
                st.markdown("---")
        else:
            st.write("No images found in PDF.")

    # Ask a question
    question = st.text_input("(Type your text question in English/Bangla)")

    if question:
        # Try to get image description from query via LLM interpretation (multi-image support)
        image_desc = get_image_descriptions_multi(pdf_data, question)
        if image_desc is not None:
            st.success(f"🖼 Image Description(s):\n\n{image_desc}")
        else:
            # Regular QA chain call
            if "last_answer" not in st.session_state or st.session_state.last_question != question:
                with st.spinner("🤔 Generating answer..."):
                    result = pdf_data["qa_chain"]({"query": question})
                    st.session_state.last_answer = result
                    st.session_state.last_question = question
            else:
                result = st.session_state.last_answer

            st.success(f"📝 উত্তর: {result['result']}")

            if st.button("📌 Show Source Chunks & Images Used"):
                source_chunks = result.get("source_documents", [])
                text_ids = []
                image_info = []
                for doc in source_chunks:
                    if doc.metadata.get("type") == "text":
                        text_ids.append((doc.metadata.get("chunk_index"), doc.page_content))
                    elif doc.metadata.get("type") == "image":
                        image_info.append((doc.metadata.get("page"), doc.metadata.get("chunk_index")))

                if text_ids:
                    st.markdown("**Text Chunks Used:**")
                    for idx, content in text_ids:
                        with st.expander(f"Show Text Chunk {idx}"):
                            st.write(content)
                else:
                    st.markdown("No text chunks used.")

                if image_info:
                    st.markdown("**Images Used (Descriptions in 'View Extracted Images and Their Descriptions')**")
                    page_img_counter = {}
                    for page, chunk_idx in image_info:
                        page_img_counter[page] = page_img_counter.get(page, 0) + 1
                        img_num = page_img_counter[page]
                        with st.expander(f"Show Image {img_num} on Page {page}"):
                            img_match = None
                            # Find the exact image on that page and chunk index
                            # (Images and image_descriptions align by index, chunk_idx should map)
                            if chunk_idx is not None and 0 <= chunk_idx < len(pdf_data["images"]):
                                img_match = pdf_data["images"][chunk_idx]["image"]
                            if img_match:
                                st.image(img_match, caption=f"Page {page} | Image {img_num}", use_container_width=True)
                else:
                    st.markdown("No images used.")


# === 8. MAIN EXECUTION === #
def main():
    st.title("📤 Upload a PDF to Start")

    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

    if uploaded_file:
        if "pdf_data" not in st.session_state:
            file_bytes = uploaded_file.read()
            st.session_state.pdf_data = process_pdf(file_bytes)

        run_streamlit_app(st.session_state.pdf_data, pdf_name=uploaded_file.name)
    else:
        st.info("👆 Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()
