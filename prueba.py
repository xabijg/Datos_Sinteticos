import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC

datasets = ["hongos","bacterias","virus"]

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

BASE_OUTPUT_DIR = "svmlineal_c_variation"

POSSIBLE_TARGET_COLS = ["Gallstone Status", "Class", "class", "Label","HONG","VIRUS","BACT"]

def run_svm_ranking_with_C_variation(dataset_name):
    print(f"\nProcesando dataset: {dataset_name}")
    input_dir = f"databases/normalized/{dataset_name}/folds/train"

    for filename in os.listdir(input_dir):
        if not (filename.startswith("train_fold") and filename.endswith(".csv")):
            continue

        input_path = os.path.join(input_dir, filename)
        df = pd.read_csv(input_path)

        # Detectar columna objetivo
        target_col = next((col for col in POSSIBLE_TARGET_COLS if col in df.columns), None)
        if target_col is None:
            print(f"No se encontró columna objetivo en {filename}")
            continue

        y = df[target_col]
        X = df.drop(columns=[target_col, "id"], errors="ignore")
        X = X.select_dtypes(include=[np.number])  # Solo variables numéricas

        for C_val in C_values:
            print(f"  Entrenando SVM Lineal con C={C_val}")
            try:
                svm = LinearSVC(penalty='l2', dual=False, C=C_val, max_iter=50000)
                svm.fit(X, y)
            except Exception as e:
                print(f"   Error al entrenar con C={C_val}: {e}")
                continue


            coefs = np.abs(svm.coef_)
            feature_importance = np.mean(coefs, axis=0) if coefs.shape[0] > 1 else coefs.flatten()

            ranking = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importance,
                "C_value": C_val
            }).sort_values(by="importance", ascending=False)

            output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name, f"C_{C_val}")
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"ranking_{filename.replace('.csv', '')}_C{C_val}.csv"
            output_path = os.path.join(output_dir, output_filename)
            ranking.to_csv(output_path, index=False)
            print(f"   Ranking guardado: {output_path}")

if __name__ == "__main__":
    for dataset in datasets:
        run_svm_ranking_with_C_variation(dataset)
