"""Une los datasets individuales en el dataset maestro (dataset_senas.csv).

Uso:
    python unir_datasets.py                 # usa los CSV de la carpeta data/
    python unir_datasets.py --desde-ramas   # además descarga los CSV de las
                                            # ramas remotas origin/dataset_*

Validaciones que hace:
  - Columnas correctas (label + 63 coordenadas)
  - Etiquetas válidas (repara la 'Ñ' corrompida por codificación)
  - Elimina filas duplicadas (frames repetidos)
  - Muestra un resumen de muestras por persona y por letra
"""
import argparse
import glob
import io
import os
import subprocess
import sys

import pandas as pd

from comun import CARPETA_DATOS, COLUMNAS_CSV, DATASET_MAESTRO, LETRAS, leer_csv

PREFIJO_RAMA = "dataset_"


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def csvs_de_ramas(remoto="origin"):
    """Devuelve [(origen, bytes_csv)] con los data/*.csv de cada rama remota dataset_*."""
    print(f"Actualizando ramas de '{remoto}' (git fetch)...")
    try:
        git("fetch", remoto, "--prune")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  No se pudo hacer git fetch ({e}). Se usarán las ramas ya descargadas.")

    ramas = git("for-each-ref", "--format=%(refname:short)",
                f"refs/remotes/{remoto}/{PREFIJO_RAMA}*").decode().split()
    resultado = []
    for rama in ramas:
        archivos = git("ls-tree", "-r", "--name-only", rama, "--", f"{CARPETA_DATOS}/").decode().split("\n")
        archivos = [a for a in archivos if a.endswith(".csv")]
        if not archivos:
            print(f"   - {rama}: sin archivos en {CARPETA_DATOS}/ (todavía no subió datos)")
        for a in archivos:
            resultado.append((f"{rama}:{a}", git("show", f"{rama}:{a}")))
    return resultado


def cargar(origen, fuente):
    df = leer_csv(fuente)
    faltan = [c for c in COLUMNAS_CSV if c not in df.columns]
    if faltan:
        print(f"   ❌ {origen}: columnas faltantes {faltan[:5]}... — se omite")
        return None
    df = df[COLUMNAS_CSV].copy()
    df = df.apply(lambda c: pd.to_numeric(c, errors='coerce') if c.name != 'label' else c)
    malas = ~df['label'].isin(LETRAS) | df.isna().any(axis=1)
    if malas.any():
        print(f"   ⚠️  {origen}: {malas.sum()} filas con etiqueta o valores inválidos — se descartan")
        df = df[~malas]
    persona = os.path.splitext(os.path.basename(origen.split(":")[-1]))[0]
    df['persona'] = persona.removeprefix("dataset_")
    return df


def main():
    p = argparse.ArgumentParser(description="Unir datasets individuales")
    p.add_argument("--desde-ramas", action="store_true",
                   help="Incluir los CSV de data/ de todas las ramas remotas dataset_*")
    p.add_argument("--remoto", default="origin")
    args = p.parse_args()

    fuentes = {}  # nombre de archivo -> (origen, fuente); evita contar 2 veces el mismo CSV
    for ruta in sorted(glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))):
        fuentes[os.path.basename(ruta)] = (ruta, ruta)
    if args.desde_ramas:
        for origen, contenido in csvs_de_ramas(args.remoto):
            nombre = os.path.basename(origen.split(":")[-1])
            if nombre in fuentes:
                print(f"   - {origen}: ya existe '{nombre}' en local, se usa la versión de la rama")
            fuentes[nombre] = (origen, io.BytesIO(contenido))

    if not fuentes:
        print(f"⚠️  No se encontraron CSV en '{CARPETA_DATOS}/'.")
        print("Cada integrante debe capturar con: python capturar_dataset.py --nombre <su_nombre>")
        sys.exit(1)

    print(f"\n📦 Archivos a unificar ({len(fuentes)}):")
    dfs = []
    for origen, fuente in fuentes.values():
        df = cargar(origen, fuente)
        if df is not None and len(df):
            print(f"   - {origen}: {len(df)} muestras")
            dfs.append(df)

    if not dfs:
        print("❌ Ningún archivo válido.")
        sys.exit(1)

    total = pd.concat(dfs, ignore_index=True)
    antes = len(total)
    total = total.drop_duplicates(subset=COLUMNAS_CSV)
    if antes - len(total):
        print(f"\n🧹 Se eliminaron {antes - len(total)} filas duplicadas (frames repetidos).")

    print("\n📊 Muestras por letra y persona:")
    tabla = pd.crosstab(total['label'], total['persona'], margins=True, margins_name="TOTAL")
    tabla = tabla.reindex([l for l in LETRAS if l in tabla.index] + ["TOTAL"])
    print(tabla.to_string())
    faltantes = [l for l in LETRAS if l not in tabla.index]
    if faltantes:
        print(f"\n⚠️  Letras sin ninguna muestra: {faltantes}")

    total.to_csv(DATASET_MAESTRO, index=False, encoding='utf-8')
    print(f"\n✅ Dataset maestro creado: '{DATASET_MAESTRO}' ({len(total)} muestras)")
    print("   Siguiente paso: python entrenar_modelo.py")


if __name__ == "__main__":
    main()
