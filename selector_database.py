import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from collections import defaultdict

datasets = ["virus","bacterias","hongos"]
selectors = ["svmlineal", "randomforest", "mutualinfo"]

def svm_lineal_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con SVM Lineal para {dataset_name}...")
    input_dir = f"databases/normalized/{dataset_name}/folds/train"
    output_dir = os.path.join("svmlineal", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"svmlineal_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["HONG","VIRUS","BACT"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

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
    input_dir = f"databases/normalized/{dataset_name}/folds/train"
    output_dir = os.path.join("randomforest", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"randomforest_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["HONG","VIRUS","BACT"] if col in df.columns), None)
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
            print(f" Ranking RF guardado en: {output_path}")

def mutual_info_feature_ranking(dataset_name):
    print(f"\n Iniciando ranking con Mutual Information para {dataset_name}...")
    input_dir = f"databases/normalized/{dataset_name}/folds/train"
    base_output_dir = os.path.join("mutualinfo", dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)

    k_values = [3, 7, 11, 15, 19]

    for filename in os.listdir(input_dir):
        if filename.startswith("train_fold") and filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)

            df = pd.read_csv(input_path)
            target_col = next((col for col in ["HONG","VIRUS","BACT"] if col in df.columns), None)
            if target_col is None:
                print(f" No se encontró columna objetivo para {dataset_name} en {filename}")
                continue

            y = df[target_col]
            X = df.drop(columns=[target_col, "id"])
            X = X.select_dtypes(include=[np.number])

            for k in k_values:
                # Crear carpeta mutualinfo/{dataset}/{k}
                k_output_dir = os.path.join(base_output_dir, f"k{k}")
                os.makedirs(k_output_dir, exist_ok=True)

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
                output_path = os.path.join(k_output_dir, output_filename)

                ranking.to_csv(output_path, index=False)
                print(f" Ranking MutualInfo (k={k}) guardado en: {output_path}")

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

        if selector_folder == "mutualinfo":
            # Recorrer subcarpetas de k por separado
            for k_dir in os.listdir(dataset_path):
                k_path = os.path.join(dataset_path, k_dir)
                if not os.path.isdir(k_path):
                    continue

                all_stats = []
                for filename in os.listdir(k_path):
                    if not filename.endswith(".csv"):
                        continue
                    file_path = os.path.join(k_path, filename)
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
                    output_stats_path = os.path.join(stats_dir, f"{dataset}_{k_dir}_summary.csv")
                    stats_df.to_csv(output_stats_path, index=False)
                    print(f" Guardado resumen para {dataset} (k={k_dir}): {output_stats_path}")
                else:
                    print(f" No se encontraron datos válidos en: {k_path}")

        else:
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

def calcular_resumen_posiciones():
    base_path = "/home/reibax/PycharmProjects/Datos_Sinteticos"

    selectors = ["svmlineal", "randomforest", "mutualinfo"]
    datasets = ["hongos", "bacterias","virus"]

    for selector in selectors:
        for dataset in datasets:
            print(f"\nProcesando: selector = {selector} - dataset = {dataset}")

            dataset_dir = os.path.join(base_path, selector, dataset)
            print(f"Buscando archivos en: {dataset_dir}")

            if not os.path.isdir(dataset_dir):
                print(f" Carpeta no encontrada: {dataset_dir}")
                continue

            if selector == "mutualinfo":
                # Recorrer subcarpetas por cada k
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

                    # Crear resumen para este k
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
                # Caso para SVM y Random Forest
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

                # Crear resumen único para el selector/dataset
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
                output_path = os.path.join(output_dir, f"{dataset}_summary.csv")


                resumen_df.to_csv(output_path, index=False)
                print(f" Resumen guardado en: {output_path}")


if __name__ == "__main__":
    for dataset_name in datasets:
        svm_lineal_feature_ranking(dataset_name)
        random_forest_feature_ranking(dataset_name)
        mutual_info_feature_ranking(dataset_name)
    calcular_resumen_posiciones()

    # Calcular estadísticas para cada selector y dataset
    resumir_ranking_selector("svmlineal", score_column="importance")
    resumir_ranking_selector("randomforest", score_column="importance")
    resumir_ranking_selector("mutualinfo", score_column="relevance_score")
    #
    # graficar_boxplots_summary("svmlineal")
    # graficar_boxplots_summary("randomforest")
    # graficar_boxplots_summary("mutualinfo")
