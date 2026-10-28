import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

NOMBRE_CSV = "dataset_senas.csv"
NOMBRE_MODELO = "modelo_senas.pkl"

print("=== FASE 3: ENTRENAMIENTO DEL MODELO DE IA ===")
df = pd.read_csv(NOMBRE_CSV)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
precision = accuracy_score(y_test, y_pred)

print(f"\n¡Modelo Entrenado con Éxito!")
print(f"Precisión global: {precision * 100:.2f}%")

with open(NOMBRE_MODELO, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo guardado como '{NOMBRE_MODELO}'.")