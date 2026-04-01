# utils/visuals.py — GPT vision captioning for extracted image crops.
# Image extraction is handled by the per-type ingestors (ingest/ingestors/).

import os
import base64
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import settings
from utils.dataclasses import ImageCrop, Caption, CaptionOutput


def caption_image_crops(image_crops: List[ImageCrop]) -> List[Caption]:
    """
    Send each ImageCrop to GPT-5.4 mini vision and return a Caption per image.

    Uses settings.CAPTION_PROMPT which asks the model for:
      - chart type
      - axis labels and units
      - key data values / ranges
      - overall trend or insight
    """
    llm = ChatOpenAI(model=settings.VISION_MODEL, max_tokens=512)
    structured_llm = llm.with_structured_output(CaptionOutput)
    captions: List[Caption] = []

    for i, crop in enumerate(image_crops):
        base64_img = base64.b64encode(crop.image_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": settings.CAPTION_PROMPT,
                },
            ]
        )

        result: CaptionOutput = structured_llm.invoke([message])
        captions.append(Caption(
            page=crop.page,
            chart_name=result.chart_name,
            description=result.description,
            source_file=crop.source_file,
        ))
        print(f"[caption_visuals] '{result.chart_name}' — "
              f"image {i + 1}/{len(image_crops)} "
              f"(page {crop.page}, {os.path.basename(crop.source_file)})")

    return captions
