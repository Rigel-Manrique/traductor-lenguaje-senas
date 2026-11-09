"""Entrena el clasificador de señas a partir del dataset maestro.

Uso:
    python entrenar_modelo.py

Requiere haber ejecutado antes: python unir_datasets.py
"""
import pickle
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from comun import (COLUMNAS_COORDS, DATASET_MAESTRO, MODELO_IA_PATH, espejar,
                   leer_csv, normalizar_landmarks)

VERSION_CARACTERISTICAS = 2  # 2 = normalizadas respecto a la muñeca + espejo


def crear_modelo():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def preparar(df, aumentar_espejo=True):
    """Normaliza y, opcionalmente, añade copias espejadas para que el modelo
    reconozca la seña hecha con mano derecha o izquierda."""
    X = normalizar_landmarks(df[COLUMNAS_COORDS].to_numpy())
    y = df['label'].to_numpy()
    if aumentar_espejo:
        X = np.vstack([X, espejar(X)])
        y = np.concatenate([y, y])
    return X, y


def main():
    print("=== ENTRENAMIENTO DEL MODELO DE IA ===")
    try:
        df = leer_csv(DATASET_MAESTRO)
    except FileNotFoundError:
        print(f"ERROR: no existe '{DATASET_MAESTRO}'. Ejecuta primero: python unir_datasets.py")
        sys.exit(1)

    df = df.drop_duplicates(subset=['label'] + COLUMNAS_COORDS)
    if 'persona' not in df.columns:
        df['persona'] = 'desconocido'
    print(f"Muestras: {len(df)} | Letras: {df['label'].nunique()} | Personas: {df['persona'].nunique()}")

    # 1) Evaluación clásica: 80/20 aleatorio
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    modelo = crear_modelo().fit(*preparar(train))
    X_test, y_test = preparar(test, aumentar_espejo=False)
    y_pred = modelo.predict(X_test)
    print(f"\nPrecisión (división aleatoria 80/20): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 2) Evaluación realista: ¿funciona con una persona que el modelo no ha visto?
    personas = sorted(df['persona'].unique())
    if len(personas) >= 2:
        print("Precisión con personas NO vistas en el entrenamiento (deja-una-persona-fuera):")
        for persona in personas:
            entrenamiento = df[df['persona'] != persona]
            prueba = df[df['persona'] == persona]
            m = crear_modelo().fit(*preparar(entrenamiento))
            Xp, yp = preparar(prueba, aumentar_espejo=False)
            print(f"   - {persona:<15} {accuracy_score(yp, m.predict(Xp)) * 100:6.2f}%  ({len(prueba)} muestras)")
        print("   (Si una persona sale baja, faltan datos parecidos a los suyos: más gente = más robusto)\n")

    # 3) Modelo final con TODOS los datos
    modelo_final = crear_modelo().fit(*preparar(df))
    paquete = {
        "modelo": modelo_final,
        "version_caracteristicas": VERSION_CARACTERISTICAS,
        "letras": sorted(df['label'].unique()),
        "personas": personas,
        "muestras": len(df),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    with open(MODELO_IA_PATH, 'wb') as f:
        pickle.dump(paquete, f)
    print(f"Modelo final guardado como '{MODELO_IA_PATH}'.")


if __name__ == "__main__":
    main()
