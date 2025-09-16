import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

# Nombre del dataset
dataset_name = "gallstone"

input_dir = f"../databases/normalized/{dataset_name}/folds"

output_dir = os.path.join("../mutualinfo", dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Lista de vecinos a usar
k_values = [3, 7, 11, 15, 19]

# Iterar sobre archivos que comienzan con "train_fold"
for filename in os.listdir(input_dir):
    if filename.startswith("train_fold") and filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)

        # Cargar datos
        df = pd.read_csv(input_path)
        y = df["Gallstone Status"]

        # Preprocesamiento: eliminar columnas no numéricas o identificadores
        X = df.drop(columns=["Gallstone Status", "id"])
        X = X.select_dtypes(include=[np.number])
        X_scaled = X  # Escalado desactivado

        for k in k_values:
            # Calcular mutual information con k vecinos
            mi = mutual_info_classif(X_scaled, y, n_neighbors=k, random_state=42)
            adjusted_score = 1 - mi

            # Crear ranking
            ranking = pd.DataFrame({
                "index": np.arange(len(X.columns)),
                "feature": X.columns,
                "mutual_info_raw": mi,
                "relevance_score": adjusted_score,
                "n_neighbors": k
            }).sort_values(by="relevance_score", ascending=False)

            # Nombre de salida
            output_filename = f"mutualinfo_{filename.replace('.csv', f'_k{k}.csv')}"
            output_path = os.path.join(output_dir, output_filename)

            # Guardar ranking
            ranking.to_csv(output_path, index=False)
            print(f"✅ Ranking guardado: {output_path}")
