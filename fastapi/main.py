from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from pydub import AudioSegment

app = FastAPI()

@app.get('/')
async def root():
    return{"message":"hi"}


# Endpoint 1: Image Upload and Contours Detection
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image from the uploaded file
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Convert the result back to an image format
    pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")

# Endpoint 2: Text to Base64 Encoding
class TextData(BaseModel):
    text: str

@app.post("/encode-base64/")
async def encode_base64(data: TextData):
    # Encode the input text to Base64
    encoded_bytes = base64.b64encode(data.text.encode('utf-8'))
    encoded_str = encoded_bytes.decode('utf-8')
    
    return {"original_text": data.text, "base64_encoded_text": encoded_str}

# Endpoint 3: Audio Fast-Forward
@app.post("/fast-forward-audio/")
async def fast_forward_audio(file: UploadFile = File(...), fast_forward_time: int = 5000):
    # Load the audio file
    audio = AudioSegment.from_file(file.file, format=file.filename.split(".")[-1])

    # Fast-forward by slicing the audio after the fast-forward time
    fast_forwarded_audio = audio[fast_forward_time:]

    # Save the fast-forwarded audio to a buffer
    buffer = BytesIO()
    fast_forwarded_audio.export(buffer, format="wav")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


