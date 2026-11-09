"""Captura de landmarks de la mano para construir el dataset de cada integrante.

Uso:
    python capturar_dataset.py --nombre juan
    python capturar_dataset.py --nombre juan --letras A B C --muestras 150

Cada persona guarda en su propio archivo: data/dataset_<nombre>.csv
Si el archivo ya existe, las nuevas muestras se AGREGAN (no se borra nada).
"""
import argparse
import csv
import os
import re
import sys
import time

import cv2
import mediapipe as mp

from comun import (CARPETA_DATOS, COLUMNAS_CSV, LETRAS, crear_detector,
                   landmarks_a_lista)

VENTANA = "Captura de Dataset"


def parse_args():
    p = argparse.ArgumentParser(description="Captura de dataset de señas por persona")
    p.add_argument("--nombre", help="Tu nombre o usuario (se usa en el nombre del archivo)")
    p.add_argument("--letras", nargs="+", default=LETRAS,
                   help="Letras a capturar (por defecto, todo el abecedario)")
    p.add_argument("--muestras", type=int, default=100, help="Muestras por letra (default 100)")
    p.add_argument("--intervalo", type=float, default=0.05,
                   help="Segundos mínimos entre muestras para evitar frames repetidos (default 0.05)")
    p.add_argument("--camara", type=int, default=0, help="Índice de la cámara (default 0)")
    return p.parse_args()


def nombre_seguro(nombre):
    nombre = nombre.strip().lower()
    nombre = re.sub(r"[^a-z0-9_ñáéíóú-]+", "_", nombre)
    return nombre.strip("_")


def main():
    args = parse_args()
    nombre = args.nombre or input("Escribe tu nombre (ej. juan): ")
    nombre = nombre_seguro(nombre)
    if not nombre:
        print("ERROR: nombre vacío.")
        sys.exit(1)

    letras = [l.upper() for l in args.letras]
    invalidas = [l for l in letras if l not in LETRAS]
    if invalidas:
        print(f"ERROR: letras no válidas: {invalidas}")
        sys.exit(1)

    os.makedirs(CARPETA_DATOS, exist_ok=True)
    ruta_csv = os.path.join(CARPETA_DATOS, f"dataset_{nombre}.csv")
    if not os.path.exists(ruta_csv):
        with open(ruta_csv, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(COLUMNAS_CSV)

    detector = crear_detector()
    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara.")
        sys.exit(1)

    print("=== RECOLECCIÓN DE DATASET ===")
    print(f"Archivo destino: {ruta_csv}")
    print(f"Letras: {letras}")
    print(f"Muestras por letra: {args.muestras}")
    print("Controles: [ESPACIO] iniciar letra | [S] saltar letra | [Q] salir guardando lo capturado")
    print("Consejo: mueve un poco la mano (distancia, giro, ambas manos) para dar variedad.\n")

    with open(ruta_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for letra in letras:
            # Esperar a que la persona esté lista
            saltar = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: no se pudo leer la cámara.")
                    sys.exit(1)
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Letra '{letra}': ESPACIO para empezar",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "S = saltar  |  Q = salir", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow(VENTANA, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    break
                if key in (ord('s'), ord('S')):
                    saltar = True
                    break
                if key in (ord('q'), ord('Q')):
                    terminar(cap, ruta_csv)
            if saltar:
                print(f"Letra '{letra}' omitida.")
                continue

            # Conteo regresivo de 3 segundos (mostrando video en vivo)
            fin = time.time() + 3
            while time.time() < fin:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                restante = int(fin - time.time()) + 1
                cv2.putText(frame, f"Prepara '{letra}' en: {restante}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(VENTANA, frame)
                cv2.waitKey(1)

            muestras = 0
            ultima = None
            ultimo_tiempo = 0.0
            while muestras < args.muestras:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))

                if resultado.hand_landmarks:
                    hand_landmarks = resultado.hand_landmarks[0]
                    h, w, _ = frame.shape
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

                    coords = landmarks_a_lista(hand_landmarks)
                    ahora = time.time()
                    # Evitar guardar el mismo frame repetido (la cámara entrega
                    # menos frames por segundo de los que procesa el bucle)
                    if coords != ultima and ahora - ultimo_tiempo >= args.intervalo:
                        writer.writerow([letra] + coords)
                        ultima = coords
                        ultimo_tiempo = ahora
                        muestras += 1

                cv2.putText(frame, f"Capturando '{letra}': {muestras}/{args.muestras}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(VENTANA, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                    f.flush()
                    terminar(cap, ruta_csv)

            f.flush()
            print(f"Letra '{letra}' capturada ({muestras} muestras).")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n¡Captura completada! Datos guardados en '{ruta_csv}'.")
    print("Sube SOLO ese archivo a tu rama (ver documentación).")


def terminar(cap, ruta_csv):
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura detenida. Lo capturado hasta ahora quedó en '{ruta_csv}'.")
    sys.exit(0)


if __name__ == "__main__":
    main()
