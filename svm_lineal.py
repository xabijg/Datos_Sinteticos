import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Ruta del archivo de entrada
input_path = "databases/normalized/DARWIN/holdout/train_DARWIN_normalized.csv"

# Ruta de salida
output_dir = "svmlineal"
os.makedirs(output_dir, exist_ok=True)

# Nombre base del archivo original
filename = os.path.basename(input_path)
output_filename = f"svmlineal_{filename}"
output_path = os.path.join(output_dir, output_filename)

# Cargar CSV
df = pd.read_csv(input_path)

# Separar variable objetivo
y = df["class"]

# Eliminar columnas no numéricas o identificadores
# Ejemplo: supongamos que hay una columna llamada 'id'
X = df.drop(columns=["class"])
X = X.select_dtypes(include=[np.number])  # mantiene solo columnas numéricas

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo SVM lineal (L2)
svm = LinearSVC(penalty='l2', dual=False, max_iter=10000)
svm.fit(X_scaled, y)

# Obtener coeficientes y calcular importancia
coefs = np.abs(svm.coef_)
if coefs.shape[0] > 1:
    feature_importance = np.mean(coefs, axis=0)
else:
    feature_importance = coefs.flatten()

# Crear ranking
ranking = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

# Guardar resultado
ranking.to_csv(output_path, index=False)

print(f"Ranking guardado en: {output_path}")
