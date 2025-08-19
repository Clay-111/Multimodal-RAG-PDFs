# =========================
# File: ui/streamlit_ui.py
# =========================

import streamlit as st

from qa.multi_image_query import get_image_descriptions_multi


def run_streamlit_app(pdf_data, pdf_name):
    st.title("📘 Multimodal Agentic RAG for PDF Q&A")
    st.markdown(f"#### {pdf_name}")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Each element: {"question": ..., "answer": ...}

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
                        (
                            desc["description"]
                            for desc in pdf_data["image_descriptions"]
                            if desc["page"] == img_data["page"]
                        ),
                        "No description available.",
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

        # Save the Q&A pair in chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "source_docs": source_docs,
        })

    # --- Display Chat History ---
    st.markdown("### 💬 Chat History")
    for idx, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")

        # Button to show source chunks & images for this Q&A
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
                            st.image(
                                img_match,
                                caption=f"Page {page} | Image {img_num}",
                                use_container_width=True,
                            )
            else:
                st.markdown("No images used.")

        st.markdown("---")

    # --- Clear Chat Button ---
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []