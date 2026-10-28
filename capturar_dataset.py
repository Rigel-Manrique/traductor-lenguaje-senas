import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuración del dataset
LETRAS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
MUESTRAS_POR_LETRA = 100
NOMBRE_ARCHIVO_CSV = "dataset_senas.csv"
MODEL_PATH = "hand_landmarker.task"

# Verificar que exista el modelo descargado
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encontró el archivo '{MODEL_PATH}'.")
    print("Ejecuta en PowerShell: Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'hand_landmarker.task'")
    exit()

# Configurar el detector de manos de la API moderna
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Crear o preparar el archivo CSV
if not os.path.exists(NOMBRE_ARCHIVO_CSV):
    with open(NOMBRE_ARCHIVO_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        encabezado = ['label']
        for i in range(21):
            encabezado.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(encabezado)

cap = cv2.VideoCapture(0)

print("=== SCRIPT DE RECOLECCIÓN DE DATASET (API MODERNA) ===")
print(f"Letras a capturar: {LETRAS}")
print(f"Muestras por letra: {MUESTRAS_POR_LETRA}")

for letra in LETRAS:
    print(f"\n---> PREPARADO PARA CAPTURAR LA LETRA: {letra} <---")
    
    # Conteo regresivo de 3 segundos
    for count in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Acomoda la mano para '{letra}' en: {count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Captura de Dataset - Prototipo", frame)
        cv2.waitKey(1000)

    muestras_guardadas = 0
    while muestras_guardadas < MUESTRAS_POR_LETRA:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear imagen en formato MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        # Si detecta la mano
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Dibujar los puntos sobre la mano en el cuadro
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Extraer coordenadas
            coordenadas = [letra]
            for lm in hand_landmarks:
                coordenadas.extend([lm.x, lm.y, lm.z])
            
            # Guardar en el CSV
            with open(NOMBRE_ARCHIVO_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(coordenadas)

            muestras_guardadas += 1

        # Mostrar progreso
        cv2.putText(frame, f"Capturando '{letra}': {muestras_guardadas}/{MUESTRAS_POR_LETRA}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Captura de Dataset - Prototipo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Captura cancelada.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"¡Letra '{letra}' capturada con éxito!")

cap.release()
cv2.destroyAllWindows()
print(f"\n¡Dataset completado! Se generó el archivo '{NOMBRE_ARCHIVO_CSV}'.")