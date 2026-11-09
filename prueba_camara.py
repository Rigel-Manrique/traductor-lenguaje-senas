"""Prueba rápida del entorno: abre la cámara y dibuja los puntos de la mano.

Usa la API moderna de MediaPipe (Tasks); la API antigua `mp.solutions` ya no
existe en las versiones recientes de mediapipe.
"""
import cv2
import mediapipe as mp

from comun import crear_detector

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),         # índice
    (5, 9), (9, 10), (10, 11), (11, 12),    # medio
    (9, 13), (13, 14), (14, 15), (15, 16),  # anular
    (13, 17), (17, 18), (18, 19), (19, 20), # meñique
    (0, 17),
]

detector = crear_detector()
cap = cv2.VideoCapture(0)

print("Abriendo cámara... Presiona 'q' en la ventana de video para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara web.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))

    if resultado.hand_landmarks:
        h, w, _ = frame.shape
        puntos = [(int(lm.x * w), int(lm.y * h)) for lm in resultado.hand_landmarks[0]]
        for a, b in CONEXIONES:
            cv2.line(frame, puntos[a], puntos[b], (255, 255, 255), 2)
        for p in puntos:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)

    cv2.imshow("Prueba del Entorno - Prototipo Señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
