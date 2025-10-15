import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from collections import defaultdict

datasets = ["bacterias2"]
selectors = ["sinselector_importances"]


def svm_lineal_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con SVM Lineal para {dataset_name}...")
    input_dir = f"databases/splits/folds/{dataset_name}"
    output_dir = os.path.join("selector", "svmlineal", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"svmlineal_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)

            # Identificamos la columna objetivo
            target_col = next((col for col in ["HONG", "VIRUS", "BACT"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])  # Quitamos columna objetivo e ID
            X = X.select_dtypes(include=[np.number])  # Solo usamos columnas numéricas

            # Entrenamos un clasificador SVM lineal para obtener los coeficientes como importancia
            svm = LinearSVC(penalty='l2', dual=False, max_iter=50000)
            svm.fit(X, y)

            coefs = np.abs(svm.coef_)
            feature_importance = np.mean(coefs, axis=0) if coefs.shape[0] > 1 else coefs.flatten()

            ranking = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importance
            }).sort_values(by="importance", ascending=False)

            ranking.to_csv(output_path, index=False)
            print(f" Ranking SVM guardado en: {output_path}")


def random_forest_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con Random Forest (ExtraTrees) para {dataset_name}...")
    input_dir = f"databases/splits/folds/{dataset_name}"
    output_dir = os.path.join("selector", "randomforest", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"randomforest_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["HONG", "VIRUS", "BACT"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            # Entrenamos un clasificador ExtraTrees y usamos las importancias como ranking
            forest = ExtraTreesClassifier(n_estimators=10000, n_jobs=-1, random_state=42) #Preguntar Placido si aqui tambien deberia probar con otros valores
            forest.fit(X, y)

            importances = forest.feature_importances_

            ranking = pd.DataFrame({
                "index": np.arange(len(X.columns)),
                "feature": X.columns,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            ranking.to_csv(output_path, index=False)
            print(f" Ranking RF guardado en: {output_path}")


def mutual_info_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con Mutual Information para {dataset_name}...")
    input_dir = f"databases/splits/folds/{dataset_name}"
    base_output_dir = os.path.join("selector", "mutualinfo", dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Probar diferentes valores de k para la estimación de la entropía
    k_values = [3, 7, 11, 15, 19]

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["HONG", "VIRUS", "BACT"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            for k in k_values:
                k_output_dir = os.path.join(base_output_dir, f"k{k}")
                os.makedirs(k_output_dir, exist_ok=True)

                # Calcular la información mutua con k vecinos
                mi = mutual_info_classif(X, y, n_neighbors=k, random_state=42)
                adjusted_score = 1 - mi  # Ajuste para convertirlo en "ranking" (menor es mejor)

                ranking = pd.DataFrame({
                    "index": np.arange(len(X.columns)),
                    "feature": X.columns,
                    "mutual_info_raw": mi,
                    "relevance_score": adjusted_score,
                    "n_neighbors": k
                }).sort_values(by="relevance_score", ascending=False)

                output_filename = f"mutualinfo_{filename.replace('.csv', f'_k{k}.csv')}"
                output_path = os.path.join(k_output_dir, output_filename)

                ranking.to_csv(output_path, index=False)
                print(f" Ranking MutualInfo (k={k}) guardado en: {output_path}")


def calcular_resumen_posiciones(dataset_name):
    # Esta función recorre los rankings generados y calcula estadísticas de posición
    base_path = os.path.join("selector")

    for selector in selectors:
        for dataset in datasets:
            print(f"\nProcesando: selector = {selector} - dataset = {dataset}")

            dataset_dir = os.path.join(base_path, selector, dataset)
            print(f"Buscando archivos en: {dataset_dir}")

            if not os.path.isdir(dataset_dir):
                print(f" Carpeta no encontrada: {dataset_dir}")
                continue

            if selector == "mutualinfo":
                for k_dir in os.listdir(dataset_dir):
                    k_path = os.path.join(dataset_dir, k_dir)
                    if not os.path.isdir(k_path):
                        continue

                    posiciones_por_feature = defaultdict(list)

                    for filename in os.listdir(k_path):
                        if not filename.endswith(".csv"):
                            continue
                        filepath = os.path.join(k_path, filename)
                        df = pd.read_csv(filepath)
                        feature_col = next((col for col in df.columns if "feature" in col.lower()), None)
                        if feature_col is None:
                            print(f" No se encontró columna 'feature' en {filename}")
                            continue
                        for idx, feature in enumerate(df[feature_col]):
                            posiciones_por_feature[feature].append(idx)

                    resumen = []
                    for feature, posiciones in posiciones_por_feature.items():
                        resumen.append({
                            "feature": feature,
                            "mean_pos": np.mean(posiciones),
                            "median_pos": np.median(posiciones),
                            "std_pos": np.std(posiciones)
                        })

                    resumen_df = pd.DataFrame(resumen)
                    resumen_df.sort_values(by="mean_pos", inplace=True)

                    output_dir = os.path.join(base_path, selector, "stats")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{dataset}_{k_dir}_summary.csv")

                    resumen_df.to_csv(output_path, index=False)
                    print(f" Resumen guardado en: {output_path}")

            else:
                posiciones_por_feature = defaultdict(list)

                for filename in os.listdir(dataset_dir):
                    if not filename.endswith(".csv"):
                        continue
                    filepath = os.path.join(dataset_dir, filename)
                    df = pd.read_csv(filepath)
                    feature_col = next((col for col in df.columns if "feature" in col.lower()), None)
                    if feature_col is None:
                        print(f" No se encontró columna 'feature' en {filename}")
                        continue
                    for idx, feature in enumerate(df[feature_col]):
                        posiciones_por_feature[feature].append(idx)

                resumen = []
                for feature, posiciones in posiciones_por_feature.items():
                    resumen.append({
                        "feature": feature,
                        "mean_pos": np.mean(posiciones),
                        "median_pos": np.median(posiciones),
                        "std_pos": np.std(posiciones)
                    })

                resumen_df = pd.DataFrame(resumen)
                resumen_df.sort_values(by="mean_pos", inplace=True)

                output_dir = os.path.join(base_path, selector, "stats")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{dataset}_resumen.csv")

                resumen_df.to_csv(output_path, index=False)
                print(f" Resumen guardado en: {output_path}")


if __name__ == "__main__":

    for dataset_name in datasets:
        #svm_lineal_feature_ranking(dataset_name)
        #random_forest_feature_ranking(dataset_name)
        #mutual_info_feature_ranking(dataset_name)
        calcular_resumen_posiciones(dataset_name)


