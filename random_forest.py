import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from mutual import X_scaled

# Ruta del archivo de entrada
input_path = "databases/normalized/gallstone/holdout/test_gallstone_normalized.csv"

# Ruta de salida
output_dir = "randomforest"
os.makedirs(output_dir, exist_ok=True)

# Nombre base del archivo original
filename = os.path.basename(input_path)
output_filename = f"randomforest_{filename}"
output_path = os.path.join(output_dir, output_filename)

# Cargar CSV
df = pd.read_csv(input_path)

# Separar variable objetivo
y = df["Gallstone Status"]

# Eliminar columnas no numéricas o identificadores
X = df.drop(columns=["Gallstone Status","id"])
X = X.select_dtypes(include=[np.number])  # solo columnas numéricas

# Escalar datos (opcional para árboles, pero mantiene consistencia con otros métodos)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

# Modelo ExtraTreesClassifier (Random Forest) con muchos árboles y uso de todos los núcleos
forest = ExtraTreesClassifier(n_estimators=10000, n_jobs=-1, random_state=42)
forest.fit(X_scaled, y)

# Obtener importancia de características
importances = forest.feature_importances_

# Crear DataFrame con índice, nombre y score
ranking = pd.DataFrame({
    "index": np.arange(len(X.columns)),
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Guardar ranking a CSV
ranking.to_csv(output_path, index=False)

print(f"Ranking de características (Random Forest) guardado en: {output_path}")
