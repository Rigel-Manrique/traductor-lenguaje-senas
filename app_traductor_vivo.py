import cv2
import time
import os
import pickle
import threading
import numpy as np
import pandas as pd
import pyttsx3
import mediapipe as mp
from config_bd import inicializar_bd, registrar_evento_traduccion
from comun import (COLUMNAS_COORDS, MODELO_IA_PATH, crear_detector,
                   landmarks_a_lista, normalizar_landmarks)

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

if not os.path.exists(MODELO_IA_PATH):
    print(f"ERROR: No se encontró '{MODELO_IA_PATH}'. Entrena primero el modelo (python entrenar_modelo.py).")
    exit()

with open(MODELO_IA_PATH, 'rb') as f:
    paquete = pickle.load(f)

if isinstance(paquete, dict):
    modelo_ia = paquete["modelo"]
    usa_normalizacion = paquete.get("version_caracteristicas", 1) >= 2
    print(f"Modelo cargado: {paquete.get('muestras')} muestras, personas: {paquete.get('personas')}")
else:
    # Modelo antiguo (coordenadas crudas). Reentrena para mejores resultados.
    modelo_ia = paquete
    usa_normalizacion = False
    print("AVISO: modelo en formato antiguo. Ejecuta 'python entrenar_modelo.py' para actualizarlo.")

detector = crear_detector()

cap = cv2.VideoCapture(0)

# Variables de la lógica de construcción de palabras
palabra_actual = ""
letra_estable = ""
tiempo_inicio_letra = 0
ultima_letra_agregada = ""  # Evita repetir la letra mientras se mantiene la seña
TIEMPO_CONFIRMACION_SEGUNDOS = 0.8  # Tiempo para confirmar la letra
UMBRAL_CONFIANZA = 70

print("\n=== TRADUCTOR EN VIVO CON VOZ ===")
print("Instrucciones:")
print(" 1. Haz clic SOBRE LA VENTANA DE VIDEO de la cámara.")
print(" 2. Mantén la seña ~1 segundo para agregar la letra a la palabra.")
print("    Para repetir una letra (ej. 'LL'), baja la mano o cambia de seña y vuelve a hacerla.")
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

        # Extraer coordenadas (misma preparación que en el entrenamiento)
        coordenadas = landmarks_a_lista(hand_landmarks)
        if usa_normalizacion:
            entrada = normalizar_landmarks(coordenadas)
        else:
            entrada = pd.DataFrame([coordenadas], columns=COLUMNAS_COORDS)

        # Inferencia
        probabilidades = modelo_ia.predict_proba(entrada)[0]
        indice = int(np.argmax(probabilidades))
        prediccion = modelo_ia.classes_[indice]
        confianza = probabilidades[indice] * 100

        # Lógica de acumulación de letras
        if confianza > UMBRAL_CONFIANZA:
            if prediccion != letra_estable:
                letra_estable = prediccion
                tiempo_inicio_letra = time.time()
            elif (prediccion != ultima_letra_agregada and
                  time.time() - tiempo_inicio_letra >= TIEMPO_CONFIRMACION_SEGUNDOS):
                palabra_actual += prediccion
                ultima_letra_agregada = prediccion
                print(f"Letra agregada: {prediccion} | Palabra actual: {palabra_actual}")
                hablar_texto(prediccion)  # Pronuncia la letra capturada
                registrar_evento_traduccion(prediccion, round(confianza, 2), round(latencia_ms, 2))
        else:
            letra_estable = ""

        # Renderizado en pantalla (OpenCV no puede dibujar la Ñ, se muestra como 'NY')
        cv2.putText(frame, f"Detectado: {prediccion.replace('Ñ', 'NY')} ({confianza:.0f}%)", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        # Sin mano: permite volver a escribir la misma letra (ej. 'LL', 'RR')
        letra_estable = ""
        ultima_letra_agregada = ""

    # Subtítulo en pantalla para ver la palabra acumulada
    cv2.rectangle(frame, (20, 400), (620, 460), (0, 0, 0), -1)
    cv2.putText(frame, f"Palabra: {palabra_actual.replace('Ñ', 'NY')}", (30, 440),
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
            palabra_actual = ""  # Limpiar la palabra tras decirla
            ultima_letra_agregada = ""
    elif key == ord('c') or key == ord('C'):
        palabra_actual = palabra_actual[:-1]
        ultima_letra_agregada = ""

cap.release()
cv2.destroyAllWindows()
