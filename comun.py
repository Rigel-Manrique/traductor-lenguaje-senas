"""Constantes y funciones compartidas por captura, entrenamiento y app en vivo.

Mantener aquí la definición de columnas y la normalización garantiza que el
modelo se entrene y se use exactamente con las mismas características.
"""
import os
import sys

import numpy as np
import pandas as pd

LETRAS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

CARPETA_DATOS = "data"
DATASET_MAESTRO = "dataset_senas.csv"
MODELO_MANO_PATH = "hand_landmarker.task"
MODELO_IA_PATH = "modelo_senas.pkl"
URL_MODELO_MANO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/1/hand_landmarker.task")

NUM_PUNTOS = 21
COLUMNAS_COORDS = [f'{eje}_{i}' for i in range(NUM_PUNTOS) for eje in ('x', 'y', 'z')]
COLUMNAS_CSV = ['label'] + COLUMNAS_COORDS


def crear_detector():
    """Crea el detector de manos de MediaPipe (API Tasks)."""
    if not os.path.exists(MODELO_MANO_PATH):
        print(f"ERROR: No se encontró '{MODELO_MANO_PATH}'. Descárgalo con (PowerShell):")
        print(f"  Invoke-WebRequest -Uri '{URL_MODELO_MANO}' -OutFile '{MODELO_MANO_PATH}'")
        sys.exit(1)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base_options = python.BaseOptions(model_asset_path=MODELO_MANO_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    return vision.HandLandmarker.create_from_options(options)


def leer_csv(ruta):
    """Lee un CSV de capturas tolerando archivos guardados en UTF-8 o en cp1252
    (los scripts antiguos escribían con la codificación por defecto de Windows)
    y repara la etiqueta 'Ñ' si quedó corrompida."""
    try:
        df = pd.read_csv(ruta, encoding='utf-8')
    except UnicodeDecodeError:
        if hasattr(ruta, 'seek'):
            ruta.seek(0)
        df = pd.read_csv(ruta, encoding='cp1252')
    df['label'] = df['label'].astype(str).str.strip()
    # Única letra no ASCII del abecedario: cualquier etiqueta rara de 1 carácter es la Ñ
    # ('�' y 'ï¿½' = carácter de reemplazo; 'Ã‘' = Ñ en UTF-8 leída como cp1252)
    df.loc[df['label'].isin(['�', 'ï¿½', 'Ã‘', 'ñ']), 'label'] = 'Ñ'
    return df


def normalizar_landmarks(coords):
    """Convierte coordenadas crudas (n, 63) en características invariantes a la
    posición de la mano en la imagen y a la distancia a la cámara:
      - origen en la muñeca (punto 0)
      - escala = distancia muñeca -> nudillo del dedo medio (punto 9)
    """
    arr = np.asarray(coords, dtype=np.float64).reshape(-1, NUM_PUNTOS, 3)
    arr = arr - arr[:, :1, :]
    escala = np.linalg.norm(arr[:, 9, :2], axis=1)
    escala[escala < 1e-6] = 1.0
    arr = arr / escala[:, None, None]
    return arr.reshape(-1, NUM_PUNTOS * 3)


def espejar(features):
    """Refleja horizontalmente características ya normalizadas (mano izq <-> der)."""
    arr = np.asarray(features, dtype=np.float64).reshape(-1, NUM_PUNTOS, 3).copy()
    arr[:, :, 0] *= -1
    return arr.reshape(-1, NUM_PUNTOS * 3)


def landmarks_a_lista(hand_landmarks):
    """Lista plana [x0, y0, z0, x1, ...] a partir de los landmarks de MediaPipe."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords
