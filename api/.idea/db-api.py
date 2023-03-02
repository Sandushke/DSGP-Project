from fastapi import FastAPI, File, UploadFile
from pymongo import MongoClient
import bson

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["mycollection"]

@app.get("/image/{name}")
async def get_image(name: str):
    # Find the document with the matching name
    document = collection.find_one({"name": name})
    # Get the image data from the document
    image_data = document["data"]
    # Convert the image data to bytes
    image_bytes = bson.binary.Binary(image_data)
    # Return the image data in the response
    return image_bytes