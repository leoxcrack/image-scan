from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Dict

app = FastAPI()

# CORS Middleware para permitir acceso desde frontend o herramientas externas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta esto si deseas restringir orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/image-scan")
async def image_scan(file: UploadFile = File(...)) -> Dict[str, str]:
    contents = await file.read()

    # Convertimos a imagen OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"tipo_detectado": "otro"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cargamos clasificadores
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Detectamos elementos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    # Clasificamos
    if len(faces) > 0:
        tipo = "rostro"
    elif len(full_bodies) > 0 or len(upper_bodies) > 0:
        tipo = "cuerpo"
    else:
        tipo = "otro"

    return {"tipo_detectado": tipo}
