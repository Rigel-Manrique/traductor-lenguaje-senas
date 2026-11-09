# 🤟 Traductor de Lenguaje de Señas en Tiempo Real con IA y Voz

Un sistema de visión por computadora y Machine Learning diseñado para traducir el abecedario del lenguaje de señas en tiempo real, construir palabras fluidas y pronunciarlas mediante síntesis de voz (*Text-to-Speech*), registrando además métricas de rendimiento en una base de datos para análisis de Business Intelligence.

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python 3.10+
* **Visión por Computadora:** OpenCV (`opencv-python`)
* **Extracción de Puntos Clave:** MediaPipe (`mediapipe`) — *Hand Landmarker Task*
* **Machine Learning:** Scikit-Learn (`scikit-learn`) — *Random Forest Classifier*
* **Manipulación de Datos:** Pandas (`pandas`) y NumPy (`numpy`)
* **Síntesis de Voz (TTS):** `pyttsx3`
* **Base de Datos:** SQLite (`sqlite3`)

---

## 📐 Arquitectura de Datos y Funcionamiento

1. **Extracción de Coordenadas:** MediaPipe detecta **21 puntos de referencia (landmarks)** en la mano por cada fotograma. Cada punto tiene coordenadas $(x, y, z)$: **63 valores** ($21 \times 3$). Los CSV guardan estas coordenadas **crudas**.
2. **Normalización (`comun.py`):** Antes de entrenar y de predecir, los puntos se trasladan para que la muñeca sea el origen y se escalan según el tamaño de la mano. Así el modelo no depende de **dónde** está la mano en la imagen ni de **qué tan lejos** está de la cámara, algo clave cuando se mezclan datos de varias personas.
3. **Ambas manos:** Al entrenar se añade una copia espejada de cada muestra, para que el modelo reconozca la seña con la mano derecha o con la izquierda.
4. **Inferencia con IA:** El vector normalizado se procesa con un **Random Forest**, que entrega la letra y su confianza.
5. **Lógica de Estabilización y Voz:**
   * La seña debe mantener una confianza $\ge 70\%$ durante ~0.8 s para confirmarse.
   * Mantener la seña **no** repite la letra; para escribir letras dobles (`LL`, `RR`) se baja la mano o se cambia de seña y se vuelve a hacer.
   * Al presionar la barra espaciadora, `pyttsx3` reproduce la palabra completa en un hilo secundario.
6. **Registro de Métricas:** Cada traducción se guarda en `sistema_senas.db` (local, no se sube a Git).

> ⚠️ **Limitación conocida:** `J`, `Z`, `Ñ` (y en LSM también `K`, `Q`, `X`) llevan **movimiento**. Este sistema clasifica una sola imagen, así que solo aprende la posición final de esas letras.

---

## 📁 Estructura del Repositorio

```text
traductor-lenguaje-senas/
├── data/                      # Un CSV por persona: dataset_<nombre>.csv
│   └── dataset_legado.csv     # Datos que ya existían antes de este flujo
├── hand_landmarker.task       # Modelo de MediaPipe para detección de manos
├── comun.py                   # Constantes y normalización compartidas
├── prueba_camara.py           # Prueba rápida de cámara + detección
├── capturar_dataset.py        # Captura de datos de una persona
├── unir_datasets.py           # Une los CSV de data/ (y de las ramas) en el maestro
├── entrenar_modelo.py         # Entrena y evalúa el modelo
├── config_bd.py               # SQLite para métricas
├── app_traductor_vivo.py      # Aplicación principal (Cámara + IA + Voz + BD)
├── requirements.txt
└── .gitignore
```

Archivos **generados** que NO se suben a Git (cada quien los regenera): `dataset_senas.csv`, `modelo_senas.pkl`, `sistema_senas.db`.

---

## 🚀 Instalación

```powershell
git clone https://github.com/Rigel-Manrique/traductor-lenguaje-senas.git
cd traductor-lenguaje-senas
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prueba_camara.py      # verifica que la cámara y la detección funcionan
```

> Si falta `hand_landmarker.task`:
> ```powershell
> Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'hand_landmarker.task'
> ```

---

## 👥 Flujo colaborativo: cada persona en su rama

Cada integrante tiene su rama `dataset_<Nombre>` en GitHub (`dataset_Gael`, `dataset_Jorge`, `dataset_Juan`, `dataset_Rigel`, `dataset_Rosado`, `dataset_rafael`). En su rama **solo sube su propio CSV**. Así nadie pisa los datos de otro y no hay conflictos de merge.

### Paso 1 — Ponerse en tu rama con el código actualizado
```powershell
git fetch origin
git checkout dataset_Juan            # tu rama
git merge origin/main                # trae la versión más reciente del código
```

### Paso 2 — Capturar tus datos
```powershell
python capturar_dataset.py --nombre juan
```
* Se guarda en `data/dataset_juan.csv`. Si lo vuelves a ejecutar, **agrega** muestras sin borrar las anteriores.
* Por cada letra: presiona **ESPACIO** para empezar, **S** para saltarla y **Q** para salir (lo capturado se conserva).
* Para capturar solo algunas letras: `python capturar_dataset.py --nombre juan --letras J Z Ñ --muestras 150`
* **Para que el modelo sea robusto:** cambia de mano a mitad de la captura, acerca y aleja la mano, gírala un poco y captura en distintos lugares e iluminaciones (varias sesiones son mejores que una sola).

### Paso 3 — Subir SOLO tu CSV a tu rama
```powershell
git add data/dataset_juan.csv
git commit -m "datos: capturas de Juan"
git push origin dataset_Juan
```
> No subas `dataset_senas.csv`, `modelo_senas.pkl` ni `sistema_senas.db` (el `.gitignore` ya los excluye).

### Paso 4 — Integración (una persona, en `main`)
No hace falta hacer merge de las ramas: el script descarga el CSV de cada rama directamente.
```powershell
git checkout main
python unir_datasets.py --desde-ramas
python entrenar_modelo.py
```
`unir_datasets.py` valida columnas y etiquetas, repara la `Ñ`, elimina duplicados y muestra una tabla de **muestras por letra y por persona** para ver quién debe capturar más de qué letra.

Para guardar los datos de todos en `main` (opcional), copia cada CSV a `data/`:
```powershell
git checkout origin/dataset_Juan -- data/dataset_juan.csv
```

---

## 🤖 Entrenamiento del Modelo

```powershell
python entrenar_modelo.py
```
Muestra dos evaluaciones:
1. **División aleatoria 80/20:** suele salir muy alta porque mezcla fotogramas casi idénticos de una misma persona.
2. **Deja-una-persona-fuera:** entrena sin una persona y prueba con ella. **Es la medida real** de qué tan bien funcionará con alguien nuevo. Si una persona sale baja, hacen falta más datos variados.

Genera `modelo_senas.pkl` (incluye el modelo, la lista de letras y las personas usadas).

---

## 🎯 Aplicación en Tiempo Real

```powershell
python app_traductor_vivo.py
```

* **Haz clic sobre la ventana de la cámara** para que funcionen las teclas.
* **Mantén la seña (~0.8 s):** la letra se agrega a la palabra.
* **[ ESPACIO ]:** pronuncia la palabra completa y la registra en `sistema_senas.db`.
* **[ C ]:** borra la última letra.
* **[ Q ]:** salir.
* En pantalla la `Ñ` aparece como `NY` (OpenCV no puede dibujar ese carácter); en la voz y en la base de datos se guarda como `Ñ`.
