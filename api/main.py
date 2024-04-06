from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import mysql.connector
import numpy as np
from PIL import Image
import tensorflow as tf
import uvicorn
import smtplib
import ssl
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Adjust methods as needed
    allow_headers=["*"],
)


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_VERSION = 1
MODEL = tf.keras.models.load_model(f'../models/{MODEL_VERSION}')

# DATABASE CONFIG
DB_CONFIG = {
    'user': 'lays',
    'password': 'pachalays',
    'host': 'localhost',
    'database': 'potato',
    'raise_on_warnings': True
}
TABLE = ("table_name")

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
    ####

    # Getting database data
    query = f"SELECT * FROM {TABLE} WHERE Name=\"{predicted_class}\""

    cursor.execute(query)
    result = cursor.fetchone()
    if result in [None, ""]:
        print("Healthy plant")
        return {
            'class': "Healthy",
            'confidence': 1,
            'name': "Healthy",
            'causes': "There are currently no identifiable causes associated with this condition.",
            'symptoms': "Patients typically do not exhibit any discernible symptoms.",
            'treatment': "As there are no apparent symptoms, no specific treatment regimen is necessary."
        }

    id, name, causes, symptoms, treatment = result
    #print(id, name, causes, symptoms, treatment)

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'name': name,
        'causes': causes,
        'symptoms': symptoms,
        'treatment': treatment
    }

if __name__ == "__main__":
    # Database connection
    cnx = mysql.connector.connect(**DB_CONFIG)
    cursor = cnx.cursor()

    uvicorn.run(app, host='localhost', port=8001)

    cnx.close()