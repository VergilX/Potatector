from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_VERSION = 1
MODEL = tf.keras.models.load_model(f'../models/{MODEL_VERSION}')


# Ping
@app.get("/")
async def ping():
    return "Server is running"


def read_file_as_image(data) -> np.ndarray:
    """ Convert image to numpy array """

    image = np.array(Image.open(BytesIO(data)))

    return image


# Use this endpoint to predict result
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # file -> numpyArray
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
            'class': predicted_class,
            'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
