# =========================
# File: processing/images.py
# =========================

import io
from config import VISION_MODEL


def describe_images_with_llm(extracted_images):
    descriptions = []
    for img_data in extracted_images:
        try:
            buffer = io.BytesIO()
            img_data["image"].save(buffer, format="PNG")
            buffer.seek(0)

            prompt = (
                "Provide a short description of this figure. Include visible text, diagrams, "
                "charts, and their meaning."
            )

            response = VISION_MODEL.generate_content(
                [prompt, {"mime_type": "image/png", "data": buffer.read()}]
            )

            desc_text = response.text.strip() if getattr(response, "text", None) else "No description generated."
            descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data.get("figure_number"),
                "description": desc_text,
            })
        except Exception as e:
            descriptions.append({
                "page": img_data["page"],
                "figure_number": img_data.get("figure_number"),
                "description": f"Error describing image: {e}",
            })
    return descriptions