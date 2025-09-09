import os
import io
import re
import fitz
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generativeai import configure, GenerativeModel
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import BaseRetriever
from pydantic import PrivateAttr
import json
import uuid  # ✅ For unique PDF IDs

# ===================== CONFIG =====================
# Google Gemini
configure(api_key="")
vision_model = GenerativeModel("gemini-2.5-pro")

# Pinecone v2
os.environ["PINECONE_API_KEY"] = ""
index_name = "rag1"  # 👈 Data will be written to this index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ===================== OCR + IMAGE EXTRACTION =====================
def extract_and_clean_text(file_bytes):
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    pages_as_images = convert_from_bytes(file_bytes, dpi=300)
    full_text = ""
    for page_img in pages_as_images:
        ocr_result = pytesseract.image_to_string(page_img, lang="ben+eng")
        full_text += ocr_result + "\n"

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
                st.warning(f"Image extraction failed on page {page_index}: {e}")

    # Match figures in text
    figure_pattern = re.compile(r"(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    figure_matches = figure_pattern.findall(full_text)
    fig_counter = 0
    for img_data in extracted_images:
        if fig_counter < len(figure_matches):
            img_data["figure_number"] = figure_matches[fig_counter]
            fig_counter += 1

    # Clean text
    text = re.sub(r"\n\s*\d+\s*\n", "\n", full_text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip(), extracted_images

# ===================== IMAGE DESCRIPTION (Gemini Vision) =====================
def describe_images_with_llm(extracted_images):
    descriptions = []
    for img_data in extracted_images:
        try:
            buffer = io.BytesIO()
            img_data["image"].save(buffer, format="PNG")
            buffer.seek(0)

            prompt = "Provide a short description of this figure. Include visible text, diagrams, charts, and their meaning."

            response = vision_model.generate_content(
                [prompt, {"mime_type": "image/png", "data": buffer.read()}]
            )

            desc_text = response.text.strip() if response.text else "No description generated."
            descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data["figure_number"],
                "description": desc_text
            })
        except Exception as e:
            descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data["figure_number"],
                "description": f"Error describing image: {e}"
            })
    return descriptions

# ===================== CHUNKING =====================
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
            "figure_number": desc["figure_number"]
        }
        image_docs.append(Document(page_content=desc["description"], metadata=metadata))
    return image_docs

# ===================== EMBEDDINGS =====================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# ===================== PINECONE UPLOAD =====================
# Minimal, surgical change: add pdf_id so each vector is uniquely namespaced and tagged
# (keeps original logic intact; only adds ID prefix + metadata field)
def upsert_to_pinecone(docs, embedder, pdf_id):
    vectors = []
    for doc in docs:
        vec = embedder.embed_query(doc.page_content)
        vectors.append({
            "id": f"{pdf_id}-{doc.metadata['type']}-{doc.metadata['chunk_index']}",
            "values": vec,
            "metadata": {**doc.metadata, "pdf_id": pdf_id, "page_content": doc.page_content}
        })
    # Also write into a namespace matching this PDF ID to allow per-PDF retrieval if desired
    index.upsert(vectors=vectors, namespace=str(pdf_id))

# ===================== CUSTOM PINECONE RETRIEVER =====================
class PineconeRetriever(BaseRetriever):
    _index: any = PrivateAttr()
    _embedder: any = PrivateAttr()
    _top_k: int = PrivateAttr()
    _namespace: str = PrivateAttr(default="")

    def __init__(self, index, embedder, top_k=5, namespace=""):
        super().__init__()
        self._index = index
        self._embedder = embedder
        self._top_k = top_k
        self._namespace = namespace

    def get_relevant_documents(self, query: str):
        q_vec = self._embedder.embed_query(query)
        results = self._index.query(
            vector=q_vec,
            top_k=self._top_k,
            include_metadata=True,
            namespace=self._namespace
        )
        docs = []
        for match in results.matches:
            content = match.metadata.get("page_content", "")  # <-- use stored content
            docs.append(Document(page_content=content, metadata=match.metadata))
        return docs

# ===================== QA CHAIN =====================
@st.cache_resource
def build_qa_chain(_embedder, namespace=""):
    retriever = PineconeRetriever(index, _embedder, namespace=namespace)
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="models/gemini-2.5-pro",
            google_api_key=""
        ),
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# ===================== PROCESS PDF =====================
@st.cache_resource
def process_pdf(file_bytes, pdf_id):
    text, images = extract_and_clean_text(file_bytes)
    img_desc = describe_images_with_llm(images)
    text_chunks = chunk_text(text)
    img_chunks = chunk_images(img_desc)
    all_chunks = text_chunks + img_chunks

    embedder = get_embeddings()
    upsert_to_pinecone(all_chunks, embedder, pdf_id)
    qa_chain = build_qa_chain(embedder, namespace=str(pdf_id))

    return {
        "text": text,
        "images": images,
        "image_descriptions": img_desc,
        "text_chunks": text_chunks,
        "image_chunks": img_chunks,
        "all_chunks": all_chunks,
        "embedder": embedder,
        "qa_chain": qa_chain,
        "pdf_id": str(pdf_id)
    }

# ===================== MULTI-IMAGE QUERY HANDLING =====================
def get_image_descriptions_multi(pdf_data, query):
    page_image_counts = {}
    for img in pdf_data["images"]:
        page = img["page"]
        page_image_counts[page] = page_image_counts.get(page, 0) + 1

    page_img_info = []
    for page in sorted(page_image_counts.keys()):
        page_img_info.append(f"Page {page} has {page_image_counts[page]} images.")
    page_img_info_str = "\n".join(page_img_info)

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

        parsed = json.loads(text)
        page_num = parsed.get("page")
        images_list = parsed.get("images", [])

        imgs_on_page = [img for img in pdf_data["images"] if img["page"] == page_num]
        descriptions = []
        for img_num in images_list:
            if isinstance(img_num, int) and 0 < img_num <= len(imgs_on_page):
                count = 0
                global_index = None
                for idx, img in enumerate(pdf_data["images"]):
                    if img["page"] == page_num:
                        count += 1
                        if count == img_num:
                            global_index = idx
                            break
                if global_index is not None and global_index < len(pdf_data["image_descriptions"]):
                    descriptions.append(
                        f"Image {img_num} on Page {page_num}:\n{pdf_data['image_descriptions'][global_index]['description']}"
                    )
        if descriptions:
            return "\n\n---\n\n".join(descriptions)
        else:
            return f"No descriptions found for the requested images on page {page_num}."
    except Exception:
        return None

# ===================== STREAMLIT APP (Chat Style) =====================
def run_streamlit_app(pdf_data, pdf_name):
    st.title("📘 PDF Question Answering (RAG) + Image Understanding (Chat Style)")
    st.markdown(f"#### Currently Processing: {pdf_name} (ID: {pdf_data['pdf_id']})")
    st.caption(f"Pinecone Index: {index_name} • Namespace: {pdf_data['pdf_id']}")

    # Initialize chat history per PDF
    if "chat_history_by_pdf" not in st.session_state:
        st.session_state.chat_history_by_pdf = {}
    if pdf_data['pdf_id'] not in st.session_state.chat_history_by_pdf:
        st.session_state.chat_history_by_pdf[pdf_data['pdf_id']] = []

    # Show all preprocessed chunks
    with st.expander("🔎 Show Preprocessed Text Chunks"):
        for i, chunk in enumerate(pdf_data["all_chunks"]):
            label = f"**{chunk.metadata.get('type').capitalize()} Chunk {i}:**"
            if chunk.metadata.get("type") == "image":
                label += f" (Page {chunk.metadata.get('page')}, Fig: {chunk.metadata.get('figure_number') or 'N/A'})"
            st.markdown(label)
            st.write(chunk.page_content)
            st.markdown("---")

    # Show extracted images and descriptions
    with st.expander("🖼 View Extracted Images and Their Descriptions"):
        if pdf_data["images"]:
            page_image_count = {}
            for img_data in pdf_data["images"]:
                page = img_data["page"]
                page_image_count[page] = page_image_count.get(page, 0) + 1
                image_num = page_image_count[page]

                st.image(img_data["image"], caption=f"Page {page} | Image {image_num}", use_container_width=True)
                with st.expander(f"Show Description for Image {image_num} on Page {page}"):
                    matching_desc = next(
                        (desc["description"] for desc in pdf_data["image_descriptions"] if desc["page"] == img_data["page"]),
                        "No description available."
                    )
                    st.markdown(matching_desc)
                st.markdown("---")
        else:
            st.write("No images found in PDF.")

    # --- Chat Input ---
    question = st.text_input("(Type your text question in English/Bangla)")

    if question:
        image_desc = get_image_descriptions_multi(pdf_data, question)

        if image_desc is not None:
            answer = f"🖼 Image Description(s):\n\n{image_desc}"
            source_docs = []
        else:
            with st.spinner("🤔 Generating answer..."):
                result = pdf_data["qa_chain"]({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])

        # Save the Q&A pair in chat history (per PDF)
        st.session_state.chat_history_by_pdf[pdf_data['pdf_id']].append({
            "question": question,
            "answer": answer,
            "source_docs": source_docs
        })

    # --- Display Chat History ---
    st.markdown("### 💬 Chat History")
    for idx, chat in enumerate(st.session_state.chat_history_by_pdf[pdf_data['pdf_id']]):
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        with st.expander(f"📌 Show Source Chunks & Images for Q&A {idx+1}"):
            source_chunks = chat.get("source_docs", [])
            text_ids = []
            image_info = []

            for doc in source_chunks:
                if doc.metadata.get("type") == "text":
                    text_ids.append((doc.metadata.get("chunk_index"), doc.page_content))
                elif doc.metadata.get("type") == "image":
                    image_info.append((doc.metadata.get("page"), doc.metadata.get("chunk_index")))

            if text_ids:
                st.markdown("**Text Chunks Used:**")
                for t_idx, content in text_ids:
                    with st.expander(f"Show Text Chunk {t_idx}"):
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
                        if chunk_idx is not None:
                            idx_img = int(chunk_idx)
                            if 0 <= idx_img < len(pdf_data["images"]):
                                img_match = pdf_data["images"][idx_img]["image"]
                        if img_match:
                            st.image(img_match, caption=f"Page {page} | Image {img_num}", use_container_width=True)
            else:
                st.markdown("No images used.")

        st.markdown("---")

    # --- Optional: Clear Chat Button (per PDF) ---
    if st.button("🧹 Clear Chat for this PDF"):
        st.session_state.chat_history_by_pdf[pdf_data['pdf_id']] = []

# ===================== MAIN =====================
def main():
    st.title("📤 Upload a PDF to Start")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    # Sidebar status/info
    with st.sidebar:
        st.markdown("### Status")
        st.caption(f"Pinecone Index in use: **{index_name}**")

    if uploaded_files:
        # Persist processed PDFs across reruns
        if "pdf_store" not in st.session_state:
            st.session_state.pdf_store = {}

        # Process each new PDF once
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.pdf_store:
                pdf_id = str(uuid.uuid4())
                status_box = st.empty()
                status_box.info(f"📄 Processing: {uploaded_file.name} (ID: {pdf_id}) …")
                file_bytes = uploaded_file.read()
                # Execute processing
                data = process_pdf(file_bytes, pdf_id)
                st.session_state.pdf_store[uploaded_file.name] = data
                status_box.success(f"✅ Finished: {uploaded_file.name} (ID: {pdf_id})")

        # Select which PDF to interact with
        selected_name = st.selectbox("📑 Select a PDF to interact with:", list(st.session_state.pdf_store.keys()))
        run_streamlit_app(st.session_state.pdf_store[selected_name], selected_name)
    else:
        st.info("👆 Please upload a PDF.")

if __name__ == "__main__":
    main()
