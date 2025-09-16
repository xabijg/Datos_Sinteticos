import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier

# Nombre del dataset
dataset_name = "gallstone"

input_dir = f"../databases/normalized/{dataset_name}/folds"

output_dir = os.path.join("../randomforest", dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre archivos que comienzan con "train_fold"
for filename in os.listdir(input_dir):
    if filename.startswith("train_fold") and filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"randomforest_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Cargar CSV
        df = pd.read_csv(input_path)

        # Separar variable objetivo
        y = df["Gallstone Status"]

        # Eliminar columnas no numéricas o identificadores
        X = df.drop(columns=["Gallstone Status", "id"])
        X = X.select_dtypes(include=[np.number])

        # (Opcional) Normalización desactivada
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        X_scaled = X

        # Entrenar modelo ExtraTreesClassifier (Random Forest)
        forest = ExtraTreesClassifier(n_estimators=10000, n_jobs=-1, random_state=42)
        forest.fit(X_scaled, y)

        # Importancia de características
        importances = forest.feature_importances_

        # Crear ranking
        ranking = pd.DataFrame({
            "index": np.arange(len(X.columns)),
            "feature": X.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Guardar resultado
        ranking.to_csv(output_path, index=False)

        print(f"✅ Ranking guardado en: {output_path}")
