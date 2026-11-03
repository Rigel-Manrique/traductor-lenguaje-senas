import glob
import os
import pandas as pd

# Nombre del archivo final consolidado
DATASET_FINAL = "dataset_senas.csv"

# Buscar todos los archivos que empiecen con 'dataset_' y terminen en '.csv'
archivos_csv = glob.glob("dataset_*.csv")

# Excluir el archivo final consolidado si ya existe en la búsqueda para no duplicar
archivos_a_unir = [f for f in archivos_csv if f != DATASET_FINAL]

if not archivos_a_unir:
    print("⚠️ No se encontraron archivos de datos individuales (ej. 'dataset_juan.csv').")
    print("Asegúrate de que los integrantes hayan guardado sus capturas con el formato 'dataset_nombre.csv'.")
else:
    print(f"📦 Archivos encontrados para unificar ({len(archivos_a_unir)}):")
    for archivo in archivos_a_unir:
        print(f" - {archivo}")
    
    # Cargar y unir todos los CSVs
    dfs = [pd.read_csv(f) for f in archivos_a_unir]
    dataset_unificado = pd.concat(dfs, ignore_index=True)
    
    # Guardar el CSV maestro
    dataset_unificado.to_csv(DATASET_FINAL, index=False)
    
    print("\n✅ ¡Éxito! Dataset consolidado creado:")
    print(f"   Archivo: '{DATASET_FINAL}'")
    print(f"   Muestras totales acumuladas: {len(dataset_unificado)}")