# =========================
# File: qa/multi_image_query.py
# =========================

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY_SECOND, GEMINI_TEXT_MODEL_NAME


def get_image_descriptions_multi(pdf_data, query):
    """
    Handle multi-image queries from a PDF using an LLM to interpret
    whether the user requested specific images from specific pages.
    """

    # Count images per page
    page_image_counts = {}
    for img in pdf_data["images"]:
        page = img["page"]
        page_image_counts[page] = page_image_counts.get(page, 0) + 1

    # Build image info string
    page_img_info = [
        f"Page {page} has {page_image_counts[page]} images."
        for page in sorted(page_image_counts.keys())
    ]
    page_img_info_str = "\n".join(page_img_info)

    # Prompt for the LLM
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

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_TEXT_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY_SECOND
    )

    try:
        response = llm.predict(prompt)
        text = response.strip()

        # Case: no specific image requested
        if text.lower().startswith("no specific image"):
            return None

        # Parse JSON response
        parsed = json.loads(text)
        page_num = parsed.get("page")
        images_list = parsed.get("images", [])

        imgs_on_page = [img for img in pdf_data["images"] if img["page"] == page_num]
        descriptions = []

        for img_num in images_list:
            if isinstance(img_num, int) and 0 < img_num <= len(imgs_on_page):
                count = 0
                global_index = None

                # Find global index for this image
                for idx, img in enumerate(pdf_data["images"]):
                    if img["page"] == page_num:
                        count += 1
                        if count == img_num:
                            global_index = idx
                            break

                if (
                    global_index is not None
                    and global_index < len(pdf_data["image_descriptions"])
                ):
                    descriptions.append(
                        f"Image {img_num} on Page {page_num}:\n"
                        f"{pdf_data['image_descriptions'][global_index]['description']}"
                    )

        if descriptions:
            return "\n\n---\n\n".join(descriptions)
        else:
            return f"No descriptions found for the requested images on page {page_num}."

    except Exception:
        return None
