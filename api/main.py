from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = ["http://localhost",
           "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/0")
CLASS_NAMES = ["Acne", "Eczema", "Melanoma Skin Cancer Nevi and Moles", "Psoriasis"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/prediction")
async def predict(
        file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    imgBatch = np.expand_dims(image, 0)

    prediction = MODEL.predict(imgBatch)

    predictedDisease = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        "class": predictedDisease,
        'Confidence level': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
