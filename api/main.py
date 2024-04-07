from fastapi import FastAPI, File, UploadFile
import mysql.connector
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import io
from tensorflow.keras.models import load_model
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Adjust methods as needed
    allow_headers=["*"],
)

CLASS_NAMES = ['Potato___Bacteria', 'Potato___Early_blight', 'Potato___Fungi', 'Potato___Late_blight', 'Potato___Pest',
               'Potato___healthy']

MODEL = load_model('/home/aswin/PycharmProjects/Potatector/model/model.h5')
MODEL.load_weights('/home/aswin/PycharmProjects/Potatector/model/easy_checkpoint')

IMAGE_SIZE = 200
CHANNELS = 3

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


def predict(image):
    image = image / 255.0
    image = np.resize(image, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    image = np.expand_dims(image, axis=0)
    preds = MODEL.predict(image)
    class_idx = np.argmax(preds[0])
    class_label = CLASS_NAMES[class_idx]
    confidence = preds[0][class_idx]
    return class_label, confidence


# Use this endpoint to predict result
@app.post("/predict")
async def predict_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read()))
    image = img_to_array(image)

    class_label, confidence = predict(image)

    # Getting database data
    query = f"SELECT * FROM {TABLE} WHERE Name=\"{class_label}\""

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

    return {
        'class': class_label,
        'confidence': float(confidence)*100,
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
