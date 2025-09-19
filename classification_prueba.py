import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from collections import defaultdict

datasets = ["DIA_trainingANDTESTset_RDKit_descriptors"]
selector = "svmlineal_c_variation"
output_folder = "classification_results_svmlineal_c_variation"

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return {"ACC": acc, "AUC": auc, "PREC": prec, "RECALL": recall, "F1SCORE": f1, "SPEC": spec, "MCC": mcc}

def get_classifiers():
    classifiers = {
        "svm_linear": SVC(kernel="linear", probability=True),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "svm_poly2": SVC(kernel="poly", degree=2, probability=True),
        "svm_poly3": SVC(kernel="poly", degree=3, probability=True),
        "svm_sigmoid": SVC(kernel="sigmoid", probability=True),
    }
    for k in [3, 7, 11, 15, 19]:
        classifiers[f"knn_k{k}"] = KNeighborsClassifier(n_neighbors=k, weights="distance")
    classifiers["extratrees"] = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    classifiers["gbforest"] = GradientBoostingClassifier(n_estimators=500, random_state=42)
    classifiers["ldc"] = LinearDiscriminantAnalysis()
    classifiers["qdc"] = QuadraticDiscriminantAnalysis()
    return classifiers

for dataset in datasets:
    print(f"\nProcesando dataset: {dataset}")
    selector_path = os.path.join(selector, dataset)

    split_files = []
    for root, _, files in os.walk(selector_path):
        for f in files:
            if f.endswith(".csv") and dataset in f:
                split_files.append(os.path.join(root, f))
    split_files = sorted(split_files)

    print(f"  Archivos encontrados: {len(split_files)}")

    # Agrupar archivos por valor de C
    files_by_c = defaultdict(list)
    for f in split_files:
        c_match = re.search(r"_C([\d.]+)", f)
        if c_match:
            c_value = c_match.group(1)
            files_by_c[c_value].append(f)

    # Procesar cada valor de C
    for c_value, files in files_by_c.items():
        print(f"\nProcesando valor C = {c_value} con {len(files)} folds")

        data_json = {
            "DATASET_NAME": dataset,
            "SELECTOR": selector,
            "C_VALUE": c_value,
            "FOLDS": []
        }

        for split_path in files:
            split_file = os.path.basename(split_path)
            print(f"  Procesando archivo de ranking: {split_file}")

            match = re.search(r"fold_?(\d+)", split_file)
            if match:
                split_id = int(match.group(1))
            else:
                print(f"    No se pudo extraer el ID del fold de {split_file}")
                continue

            fold_data = {
                "FOLD": split_id,
                "DATA": []
            }

            ranking_df = pd.read_csv(split_path)
            print(f"    Ranking cargado con {len(ranking_df)} features.")

            # Preparar paths para datos train y test
            base_name = os.path.basename(split_path)
            base_name2 = base_name.replace("_train_", "_val_")
            base_name_clean = re.sub(r"_C[\d.]+(?=\.csv)", "", base_name)
            base_name2_clean = re.sub(r"_C[\d.]+(?=\.csv)", "", base_name2)
            base_name_clean = base_name_clean.replace("ranking_", "")
            base_name2_clean = base_name2_clean.replace("ranking_", "")

            train_path = f"databases/normalized/{dataset}/folds/train/{base_name_clean}"
            test_path = f"databases/normalized/{dataset}/folds/val/{base_name2_clean}"

            try:
                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)
            except FileNotFoundError as e:
                print(f"    Archivo no encontrado: {e}")
                continue

            target_col = next((col for col in ["Gallstone Status", "Class", "class", "Label"] if col in df_train.columns), None)
            if not target_col:
                print(f"    No se encontró columna objetivo en {train_path}")
                continue

            y_train = df_train[target_col]
            y_test = df_test[target_col]
            X_train_all = df_train.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number])
            X_test_all = df_test.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number])

            for clf_name, clf in get_classifiers().items():
                print(f"    Clasificador: {clf_name}")
                clf_data = {"CLASSF_NAME": clf_name, "DATA": []}

                for n in range(1, len(ranking_df) + 1):
                    selected_features = ranking_df["feature"].iloc[:n].values
                    X_train = X_train_all[selected_features]
                    X_test = X_test_all[selected_features]

                    try:
                        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
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

                        clf_data["DATA"].append({
                            "N_FEAT": n,
                            "NAME_FEAT": selected_features.tolist(),
                            "METRICS": {
                                "TRAIN": metrics_train,
                                "VAL": metrics_val
                            }
                        })

                    except Exception as e:
                        print(f"      Error con {clf_name} n_feats={n}: {e}")
                        continue

                fold_data["DATA"].append(clf_data)

            data_json["FOLDS"].append(fold_data)

        os.makedirs(output_folder, exist_ok=True)
        output_path = f"{output_folder}/{dataset}_{selector}_C{c_value}_classification.json"
        with open(output_path, "w") as f:
            json.dump(data_json, f, indent=4)
        print(f"\n  Resultados guardados en: {output_path}")