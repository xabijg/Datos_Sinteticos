import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import KMeansSMOTE


datasets = ["bacterias2"]
selectors = ["sinselector_importances"]
metrics = ["ACC", "AUC", "PREC", "RECALL", "F1SCORE", "SPEC", "MCC"]


FEATURE_LIMITS = {
    "svm_linear": 25,
    "svm_rbf": 25,
    "svm_poly2": 25,
    "svm_poly3": 25,
    "svm_sigmoid": 25,
    "extratrees": 25,
    "gbforest": 25,
    "ldc": 25,
    "qdc": 25,
    "knn_k3": 25,
    "knn_k7": 25,
    "knn_k11": 25,
    "knn_k15": 25,
    "knn_k19": 25,
}

SAVE_INTERVAL = 10


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    return {
        "ACC": acc, "AUC": auc, "PREC": prec,
        "RECALL": recall, "F1SCORE": f1, "SPEC": spec, "MCC": mcc
    }


def get_classifiers():
     classifiers = {
         #"svm_linear": lambda: SVC(kernel="linear", probability=True),
         #"svm_rbf": lambda: SVC(kernel="rbf", probability=True),
         #"svm_poly2": lambda: SVC(kernel="poly", degree=2, probability=True),
         #"svm_poly3": lambda: SVC(kernel="poly", degree=3, probability=True),
         #"svm_sigmoid": lambda: SVC(kernel="sigmoid", probability=True),
         "extratrees": lambda: ExtraTreesClassifier(
             n_estimators=500,
             max_depth=10,
             min_samples_split=10,
             min_samples_leaf=5,
             max_features="sqrt",
             n_jobs=-1,
             random_state=42)

         # "gbforest": lambda: GradientBoostingClassifier(
         #     n_estimators = 150,
         #     learning_rate = 0.05,
         #     max_depth = 2,
         #     min_samples_split = 40,
         #     min_samples_leaf = 20,
         #     subsample = 0.7,
         #     random_state = 42),
         #"ldc": lambda: LinearDiscriminantAnalysis(),
         #"qdc": lambda: QuadraticDiscriminantAnalysis()
     }

     # for k in [3, 7, 11, 15, 19]:
     #     classifiers[f"knn_k{k}"] = lambda k=k: KNeighborsClassifier(n_neighbors=k, weights="distance")
     return classifiers




for dataset in datasets:
    print(f"\nProcesando dataset: {dataset}")

    for selector in selectors:
        print(f" Selector: {selector}")
        selector_path = os.path.join("selector", selector, dataset)
        print(f"  Ruta del selector: {selector_path}")

        data_json = {
            "DATASET_NAME": dataset,
            "SELECTOR": selector,
            "DATA": []
        }

        split_files = sorted([
            os.path.join(selector_path, f)
            for f in os.listdir(selector_path)
            if f.endswith(".csv")
        ])

        print(f"  Archivos encontrados: {len(split_files)}")
        for f in split_files:
            print(f"   - {f}")

        classifiers = get_classifiers()

        for clf_name, clf_constructor in classifiers.items():
            print(f"  Clasificador: {clf_name}")
            clf_data = {"CLASSF_NAME": clf_name, "FOLDS": []}
            max_feats = FEATURE_LIMITS.get(clf_name, None)

            for split_path in split_files:
                split_file = os.path.basename(split_path)
                print(f"   Procesando archivo de ranking: {split_file}")

                match = re.search(r"fold_?(\d+)", split_file)
                if not match:
                    print(f"    No se pudo extraer el ID del fold de {split_file}")
                    continue
                split_id = int(match.group(1))
                print(f"    Fold detectado: {split_id}")

                ranking_df = pd.read_csv(split_path)
                print(f"    Ranking cargado con {len(ranking_df)} features.")

                base_name = f"{selector}_train_fold_{split_id}_{dataset}.csv"
                base_name2 = f"{selector}_val_fold_{split_id}_{dataset}.csv"
                fold_name = base_name.replace(f"{selector}_", "")
                fold_name2 = base_name2.replace(f"{selector}_", "")

                train_path = f"databases/splits/folds/{dataset}/{fold_name}"
                test_path = f"databases/splits/folds/{dataset}/{fold_name2}"

                print(f"    Train path: {train_path}")
                print(f"    Test path: {test_path}")

                try:
                    df_train = pd.read_csv(train_path)
                    df_test = pd.read_csv(test_path)
                except FileNotFoundError as e:
                    print(f"    Archivo no encontrado: {e}")
                    continue

                target_col = next((col for col in ["HONG", "BACT", "VIRUS"] if col in df_train.columns), None)
                if not target_col:
                    print(f"    No se encontró columna objetivo en {train_path}")
                    continue
                print(f"    Columna objetivo detectada: {target_col}")

                y_train_orig = df_train[target_col].copy()
                y_test = df_test[target_col]

                X_train_all_orig = df_train.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number]).copy()
                X_test_all = df_test.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number])

                split_data = {"SPLIT_ID": split_id, "DATA": []}

                for n in range(1, len(ranking_df) + 1):
                    if max_feats is not None and n > max_feats:
                        print(f"    Límite de features alcanzado para {clf_name}: {max_feats}")
                        break

                    selected_features = ranking_df["feature"].iloc[:n].values
                    last_added_feature = ranking_df["feature"].iloc[n - 1]

                    try:
                        # Copias limpias para cada iteración
                        X_train = X_train_all_orig[selected_features].copy()
                        y_train = y_train_orig.copy()

                        # Aplicar KMeans-SMOTE solo en entrenamiento
                        try:
                            smote = KMeansSMOTE(random_state=0, n_jobs=-1)
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                        except Exception as smote_error:
                            print(f"    Error aplicando KMeans-SMOTE en fold {split_id}, n_feats={n}: {smote_error}")
                            continue

                        X_test = X_test_all[selected_features]

                        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

                        clf = clf_constructor()
                        fit_params = {}
                        if 'sample_weight' in clf.fit.__code__.co_varnames:
                            fit_params['sample_weight'] = sample_weights

                        clf.fit(X_train, y_train, **fit_params)

                        y_train_pred = clf.predict(X_train)
                        y_test_pred = clf.predict(X_test)
                        y_train_proba = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None
                        y_test_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

                        metrics_train = compute_metrics(y_train, y_train_pred, y_train_proba)
                        metrics_val = compute_metrics(y_test, y_test_pred, y_test_proba)

                        split_data["DATA"].append({
                            "N_FEAT": n,
                            "NAME_FEAT": last_added_feature,
                            "DATA": {
                                "TRAIN": metrics_train,
                                "VAL": metrics_val
                            }
                        })

                        if n % SAVE_INTERVAL == 0 or n == len(ranking_df) or (max_feats and n == max_feats):
                            print(f"    Guardando progreso parcial en feature {n}...")
                            os.makedirs("classification_results", exist_ok=True)
                            temp_output = f"classification_results/tmp_{dataset}_{selector}_{clf_name}_split{split_id}.json"
                            with open(temp_output, "w") as f:
                                json.dump({
                                    "CLASSF_NAME": clf_name,
                                    "SPLIT_ID": split_id,
                                    "DATA": split_data
                                }, f, indent=4)

                    except Exception as e:
                        print(f"    Error en fold {split_id}, n_feats={n}: {e}")
                        continue

                clf_data["FOLDS"].append(split_data)

            data_json["DATA"].append(clf_data)

        os.makedirs("classification_results", exist_ok=True)
        output_path = f"classification_results/{dataset}_{selector}_classification.json"
        with open(output_path, "w") as f:
            json.dump(data_json, f, indent=4)

        print(f"Resultados guardados en: {output_path}")
