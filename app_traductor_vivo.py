import cv2
import time
import os
import pickle
import threading
import numpy as np
import pandas as pd
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config_bd import inicializar_bd, registrar_evento_traduccion

# Inicializar Base de Datos
inicializar_bd()

def hablar_texto(texto):
    """Función de voz aislada para evitar bloqueos en Windows"""
    def _hablar():
        try:
            engine_local = pyttsx3.init()
            engine_local.setProperty('rate', 140) # Velocidad del habla
            
            # Buscar voz en español
            voices = engine_local.getProperty('voices')
            for voice in voices:
                if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                    engine_local.setProperty('voice', voice.id)
                    break
                    
            engine_local.say(texto)
            engine_local.runAndWait()
        except Exception as e:
            print("Error en reproduccion de audio:", e)
    
    t = threading.Thread(target=_hablar)
    t.daemon = True
    t.start()

MODEL_PATH = "hand_landmarker.task"
MODELO_IA_PATH = "modelo_senas.pkl"

if not os.path.exists(MODELO_IA_PATH):
    print(f"ERROR: No se encontró '{MODELO_IA_PATH}'. Entrena primero el modelo.")
    exit()

with open(MODELO_IA_PATH, 'rb') as f:
    modelo_ia = pickle.load(f)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Variables de la lógica de construcción de palabras
palabra_actual = ""
letra_estable = ""
tiempo_inicio_letra = 0
TIEMPO_CONFIRMACION_SEGUNDOS = 0.5  # Tiempo para confirmar la letra

print("\n=== TRADUCTOR EN VIVO CON VOZ ===")
print("Instrucciones:")
print(" 1. Haz clic SOBRE LA VENTANA DE VIDEO de la cámara.")
print(" 2. Mantén la seña 1 segundo para agregar la letra a la palabra.")
print(" 3. Presiona [ESPACIO] para pronunciar la palabra completa.")
print(" 4. Presiona [C] para borrar la última letra.")
print(" 5. Presiona [Q] para salir.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    inicio_tiempo = time.time()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)
    latencia_ms = (time.time() - inicio_tiempo) * 1000

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Dibujar puntos
        h, w, _ = frame.shape
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Extraer coordenadas
        coordenadas = []
        for lm in hand_landmarks:
            coordenadas.extend([lm.x, lm.y, lm.z])

        cols_ordenadas = []
        for i in range(21):
            cols_ordenadas.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
            
        df_punto = pd.DataFrame([coordenadas], columns=cols_ordenadas)

        # Inferencia
        prediccion = modelo_ia.predict(df_punto)[0]
        probabilidades = modelo_ia.predict_proba(df_punto)
        confianza = np.max(probabilidades) * 100

        # Lógica de acumulación de letras
        if confianza > 70:
            if prediccion == letra_estable:
                tiempo_transcurrido = time.time() - tiempo_inicio_letra
                if tiempo_transcurrido >= TIEMPO_CONFIRMACION_SEGUNDOS:
                    palabra_actual += prediccion
                    print(f"Letra agregada: {prediccion} | Palabra actual: {palabra_actual}")
                    hablar_texto(prediccion) # Pronuncia la letra capturada
                    registrar_evento_traduccion(prediccion, round(confianza, 2), round(latencia_ms, 2))
                    letra_estable = "" 
            else:
                letra_estable = prediccion
                tiempo_inicio_letra = time.time()

        # Renderizado en pantalla
        cv2.putText(frame, f"Detectado: {prediccion} ({confianza:.0f}%)", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Subtítulo en pantalla para ver la palabra acumulada
    cv2.rectangle(frame, (20, 400), (620, 460), (0, 0, 0), -1)
    cv2.putText(frame, f"Palabra: {palabra_actual}", (30, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Traductor con Voz - Prototipo BI", frame)

    # Captura de teclas de control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == 32:  # 32 es el código ASCII para la barra espaciadora
        if palabra_actual.strip():
            print(f"---> PRONUNCIANDO PALABRA COMPLETA: {palabra_actual} <---")
            hablar_texto(palabra_actual)
            registrar_evento_traduccion(f"PALABRA: {palabra_actual}", 100.0, round(latencia_ms, 2))
            palabra_actual = "" # Limpiar la palabra tras decirla
    elif key == ord('c') or key == ord('C'):
        palabra_actual = palabra_actual[:-1]

cap.release()
cv2.destroyAllWindows()