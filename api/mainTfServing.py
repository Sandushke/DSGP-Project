from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
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

endpoint = "http://localhost:8501/v1/models/skinDisease_model:predict"
diseaseType = ["Acne", "Eczema", "Melanoma Skin Cancer Nevi and Moles", "Psoriasis"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    imgBatch = np.expand_dims(image, 0)

    json_data = {"instances": imgBatch.tolist()}

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predictedClass = diseaseType[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "Disease predicted": predictedClass,
        "Confidence level": float(confidence),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
