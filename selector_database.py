import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

datasets = ["gallstone", "DARWIN", "toxicity", "DIA_trainingANDTESTset_RDKit_descriptors"]

def svm_lineal_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con SVM Lineal para {dataset_name}...")
    input_dir = f"databases/normalized/{dataset_name}/folds"
    output_dir = os.path.join("svmlineal", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"svmlineal_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)
            # Cambiar nombre de la columna objetivo si varía según dataset
            target_col = next((col for col in ["Gallstone Status", "Class", "class", "Label"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            # MIRAR LAS ITERACIONES SIN SON SUFICIENTES

            svm = LinearSVC(penalty='l2', dual=False, max_iter=50000)
            svm.fit(X, y)

            coefs = np.abs(svm.coef_)
            feature_importance = np.mean(coefs, axis=0) if coefs.shape[0] > 1 else coefs.flatten()

            ranking = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importance
            }).sort_values(by="importance", ascending=False)

            ranking.to_csv(output_path, index=False)
            print(f"✅ Ranking SVM guardado en: {output_path}")

def random_forest_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con Random Forest (ExtraTrees) para {dataset_name}...")
    input_dir = f"databases/normalized/{dataset_name}/folds"
    output_dir = os.path.join("randomforest", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"randomforest_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["Gallstone Status", "Class", "class", "Label"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            forest = ExtraTreesClassifier(n_estimators=10000, n_jobs=-1, random_state=42)
            forest.fit(X, y)

            importances = forest.feature_importances_

            ranking = pd.DataFrame({
                "index": np.arange(len(X.columns)),
                "feature": X.columns,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            ranking.to_csv(output_path, index=False)
            print(f"✅ Ranking RF guardado en: {output_path}")

def mutual_info_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con Mutual Information para {dataset_name}...")
    input_dir = f"databases/normalized/{dataset_name}/folds"
    output_dir = os.path.join("mutualinfo", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    k_values = [3, 7, 11, 15, 19]

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["Gallstone Status", "Class", "class", "Label"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            for k in k_values:
                mi = mutual_info_classif(X, y, n_neighbors=k, random_state=42)
                adjusted_score = 1 - mi

                ranking = pd.DataFrame({
                    "index": np.arange(len(X.columns)),
                    "feature": X.columns,
                    "mutual_info_raw": mi,
                    "relevance_score": adjusted_score,
                    "n_neighbors": k
                }).sort_values(by="relevance_score", ascending=False)

                output_filename = f"mutualinfo_{filename.replace('.csv', f'_k{k}.csv')}"
                output_path = os.path.join(output_dir, output_filename)

                ranking.to_csv(output_path, index=False)
                print(f"✅ Ranking MutualInfo (k={k}) guardado en: {output_path}")

def resumir_ranking_selector(selector_folder, score_column="importance"):
    print(f"\n Calculando estadísticas para selector: {selector_folder}")
    base_path = selector_folder
    stats_dir = os.path.join(base_path, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_path):
            print(f" No existe carpeta para dataset {dataset} en {base_path}")
            continue

        all_stats = []

        for filename in os.listdir(dataset_path):
            if not filename.endswith(".csv"):
                continue
            file_path = os.path.join(dataset_path, filename)
            df = pd.read_csv(file_path)

            if score_column not in df.columns:
                continue

            stats = {
                "file": filename,
                "mean": df[score_column].mean(),
                "median": df[score_column].median(),
                "std": df[score_column].std()
            }
            all_stats.append(stats)

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            output_stats_path = os.path.join(stats_dir, f"{dataset}_summary.csv")
            stats_df.to_csv(output_stats_path, index=False)
            print(f" Guardado resumen para {dataset}: {output_stats_path}")
        else:
            print(f" No se encontraron datos válidos en: {dataset_path}")

def graficar_boxplots_summary(selector_folder):
    print(f"\n Generando boxplots para selector: {selector_folder}")

    stats_dir = os.path.join(selector_folder, "stats")
    if not os.path.isdir(stats_dir):
        print(f" No existe carpeta de estadísticas: {stats_dir}")
        return

    for summary_file in os.listdir(stats_dir):
        if not summary_file.endswith("_summary.csv"):
            continue

        summary_path = os.path.join(stats_dir, summary_file)
        df = pd.read_csv(summary_path)

        dataset_name = summary_file.replace("_summary.csv", "")

        fig, ax = plt.subplots()
        df_to_plot = df[["mean", "median", "std"]]
        df_to_plot.boxplot(ax=ax)
        ax.set_title(f"Distribución de {dataset_name} ({selector_folder})")
        ax.set_ylabel("Valor")
        ax.grid(True)

        plot_filename = f"{dataset_name}_boxplot.png"
        plot_path = os.path.join(stats_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f" Boxplot guardado: {plot_path}")

if __name__ == "__main__":
    for dataset_name in datasets:
        svm_lineal_feature_ranking(dataset_name)
        random_forest_feature_ranking(dataset_name)
        mutual_info_feature_ranking(dataset_name)

    # Calcular estadísticas para cada selector y dataset
    resumir_ranking_selector("svmlineal", score_column="importance")
    resumir_ranking_selector("randomforest", score_column="importance")
    resumir_ranking_selector("mutualinfo", score_column="relevance_score")

    graficar_boxplots_summary("svmlineal")
    graficar_boxplots_summary("randomforest")
    graficar_boxplots_summary("mutualinfo")
