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

1. **Extracción de Coordenadas:** MediaPipe detecta **21 puntos de referencia (landmarks)** en la mano por cada fotograma. Cada punto contiene coordenadas tridimensionales $(x, y, z)$, lo que genera un vector de **63 características** ($21 \times 3$).
2. **Inferencia con IA:** El vector de 63 coordenadas se procesa a través de un modelo **Random Forest**, evaluando la confianza probabilística de la predicción.
3. **Lógica de Estabilización y Voz:**
   * La seña debe mantener una confianza $\ge 70\%$ de forma continua durante un tiempo prefijado (ej. 0.6s - 1.0s) para confirmarse como una letra válida.
   * Las letras confirmadas se agregan al acumulador de palabras en pantalla.
   * Al presionar la barra espaciadora, el motor `pyttsx3` reproduce la palabra completa en un hilo secundario (`threading`) para no congelar la captura de video.
4. **Registro de Métricas:** Cada traducción y evento se guarda en `sistema_senas.db` para evaluar latencias y porcentajes de precisión.

---

## 📁 Estructura del Repositorio

```text
lenguaje_senas/
├── data/                      # Carpeta para datasets individuales de cada desarrollador
├── hand_landmarker.task       # Modelo de MediaPipe para detección de manos
├── capturar_dataset.py        # Script para recolección colaborativa de datos (27 letras)
├── unir_datasets.py           # Script para fusionar múltiples archivos CSV en un máster
├── entrenar_modelo.py         # Script para entrenar la IA con Scikit-Learn
├── config_bd.py               # Módulo de inicialización y escritura en SQLite
├── app_traductor_vivo.py      # Aplicación principal (Cámara + IA + Voz + DB)
├── requirements.txt           # Dependencias del proyecto
└── .gitignore                 # Archivos omitidos en la sincronización de Git
```

---

## 🚀 Guía de Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/traductor-lenguaje-senas.git
cd traductor-lenguaje-senas
```

### 2. Crear y Activar el Entorno Virtual (`venv`)

* **En Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **En macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

> **Nota:** El archivo `hand_landmarker.task` debe estar ubicado en la raíz del proyecto. Si no existe, puedes descargarlo con el siguiente comando en PowerShell:
> ```powershell
> Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'hand_landmarker.task'
> ```

---

## 👥 Flujo de Trabajo Colaborativo (Git + Datasets)

Para lograr que el modelo sea **ambidiestro, robusto y multiusuario**, cada integrante del equipo debe aportar capturas utilizando sus propias manos bajo distintas condiciones de iluminación y distancia.

### Paso 1: Captura de Datos Individual
Cada desarrollador debe ejecutar en su máquina:
```bash
python capturar_dataset.py
```
* **Abecedario soportado:** 27 letras (`A-Z` incluyendo la `Ñ`).
* **Recomendación:** Capturar la mitad de las muestras con la mano izquierda y la otra mitad con la mano derecha. Mover ligeramente la mano durante el conteo para capturar variaciones de distancia e inclinación.

### Paso 2: Guardar y Subir el CSV Personalizado
1. Renombra el archivo generado `dataset_senas.csv` con tu nombre o usuario de Git (ejemplo: `data/dataset_juan.csv`).
2. Sube únicamente tu CSV individual a Git:
   ```bash
   git add data/dataset_juan.csv
   git commit -m "feat: Agregar capturas de datos de Juan (Mano izq/der)"
   git push origin tu-rama
   ```

### Paso 3: Unificación de Datasets (Líder del Proyecto / Integración)
Una vez reunidos los CSVs de todo el equipo en la carpeta `data/`:
```bash
python unir_datasets.py
```
Este script concatenará todos los datasets individuales y generará el archivo `dataset_senas.csv` maestro listo para el entrenamiento.

---

## 🤖 Entrenamiento del Modelo de IA

Con el archivo `dataset_senas.csv` maestro listo, ejecuta el script de entrenamiento:

```bash
python entrenar_modelo.py
```

* **Salida esperada:** La terminal mostrará el porcentaje de precisión (*Accuracy Score*) global y generará el binario `modelo_senas.pkl`.

---

## 🎯 Ejecución de la Aplicación en Tiempo Real

Para iniciar el traductor en vivo con voz:

```bash
python app_traductor_vivo.py
```

### 🎮 Controles de la Aplicación:
* **Haz clic sobre la ventana de la cámara** para asegurar el foco de los comandos por teclado.
* **Mantén la seña fija (~0.6s - 1s):** La letra se agregará automáticamente al acumulador.
* **[ BARRA ESPACIADORA ]:** Pronuncia la palabra completa acumulada en voz alta mediante síntesis TTS y registra el evento en la base de datos `sistema_senas.db`.
* **Tecla [ C ]:** Borra la última letra si te equivocaste al deletrear.
* **Tecla [ Q ]:** Cierra la aplicación y libera los recursos de la cámara.