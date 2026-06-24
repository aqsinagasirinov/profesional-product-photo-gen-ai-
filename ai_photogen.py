"""
FastAPI endpoint - OpenAI image-to-image (edits) servisi.

Sən şəkil + prompt göndərirsən, OpenAI həmin şəkli prompt-a əsasən
redaktə edib geri qaytarır.

Endpoint: /v1/images/edits istifadə olunur (generations YOX,
çünki generations yalnız mətndən şəkil yaradır, input şəkli qəbul etmir).

Run:
    pip install fastapi uvicorn openai python-multipart python-dotenv
    python -m uvicorn ai_photogen:app --reload --port 8000

    (.env faylında OPENAI_API_KEY=sk-... sətri olmalıdır, eyni qovluqda)

Test:
    curl -X POST "http://127.0.0.1:8000/edit-image" \
         -F "image=@input.png" \
         -F "prompt=Make the sky purple and add stars" \
         -o output.png
"""

import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

# .env faylını oxuyur (eyni qovluqdaki OPENAI_API_KEY=... sətrini yaddaşa yükləyir)
# BU SƏTIR client = OpenAI(...) sətrindən ƏVVƏL olmalıdır, əks halda key tapılmır.
load_dotenv()

app = FastAPI(title="Image Edit Service")

FIXED_PROMPT = "Remove the background completely and replace it with a pure white background (#FFFFFF, seamless studio background). Keep the product 100 identical to the original — exact same shape, color, texture, material, logo, text, proportions, and details. Do not alter, redesign, stylize, or reinterpret the product in any way.Remove any hands, fingers, or human body parts holding or touching the product. Reconstruct the hidden parts of the product naturally and seamlessly, as if no hand was ever there.Center the product perfectly in the frame, with balanced empty space on all sides (standard e-commerce product shot composition).Add soft, even, professional studio lighting with a subtle natural shadow or soft reflection beneath the product to ground it realistically — no harsh shadows, no dramatic lighting.The final image must look like a high-end e-commerce product photo: sharp focus, clean edges, no background clutter, no extra objects, no text overlays, no watermarks."

FIXED_SIZE = "1024x1024"

# API key artıq .env faylından avtomatik oxunur - heç vaxt kodun içində hardcode etmə.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.post("/edit-image")
async def edit_image(
    image: UploadFile = File(...),
  
):
    """
    image  -> redaktə olunacaq input şəkli (PNG tövsiyə olunur)
    prompt -> şəklə nə ediləcəyini izah edən mətn
    size   -> 1024x1024 / 1024x1536 / 1536x1024 (gpt-image-1 üçün)
    """

    # Fayl tipini yoxla
    if image.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Dəstəklənməyən fayl tipi: {image.content_type}",
        )

    image_bytes = await image.read()

    try:
        result = client.images.edit(
            model="gpt-image-2",  # düzəliş: "gpt-image-2" mövcud deyil
            image=(image.filename, image_bytes, image.content_type),
            prompt=FIXED_PROMPT,
            size=FIXED_SIZE,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI xətası: {e}")

    # gpt-image-1 nəticəni base64 olaraq qaytarır (URL yox)
    b64_data = result.data[0].b64_json
    image_data = base64.b64decode(b64_data)

    return StreamingResponse(
        BytesIO(image_data),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=result.png"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}