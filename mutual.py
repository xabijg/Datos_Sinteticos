import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Ruta del archivo de entrada
input_path = "databases/normalized/DARWIN/holdout/train_DARWIN_normalized.csv"

# Carpeta de salida
output_dir = "mutualinfo"
os.makedirs(output_dir, exist_ok=True)

# Nombre base del archivo original
filename = os.path.basename(input_path)
output_filename = f"mutualinfo_{filename}"
output_path = os.path.join(output_dir, output_filename)

# Cargar datos
df = pd.read_csv(input_path)
y = df["class"]

# Eliminar columnas no numéricas o identificadores
X = df.drop(columns=["class"])
X = X.select_dtypes(include=[np.number])

# Escalar (por consistencia)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular mutual information con k=3
#mi = mutual_info_classif(X_scaled, y, n_neighbors=3, random_state=42)
#mi = mutual_info_classif(X_scaled, y, n_neighbors=7, random_state=42)
#mi = mutual_info_classif(X_scaled, y, n_neighbors=11, random_state=42)
#mi = mutual_info_classif(X_scaled, y, n_neighbors=15, random_state=42)
mi = mutual_info_classif(X_scaled, y, n_neighbors=19, random_state=42)

# Invertir el score: 1 - mi ⇒ mayor puntuación = más relevante
adjusted_score = 1 - mi

# Verificamos: ¿A mayor adjusted_score → más relevante?
ranking = pd.DataFrame({
    "index": np.arange(len(X.columns)),
    "feature": X.columns,
    "mutual_info_raw": mi,
    "relevance_score": adjusted_score
}).sort_values(by="relevance_score", ascending=False)

# Guardar CSV
ranking.to_csv(output_path, index=False)

print(f"Ranking invertido (1 - MI) guardado en: {output_path}")
