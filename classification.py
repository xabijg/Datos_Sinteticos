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
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

# datasets = ["gallstone", "DARWIN", "toxicity", "DIA_trainingANDTESTset_RDKit_descriptors"]
datasets = ["diabetes"]
# selectors = ["svmlineal", "randomforest", "mutualinfo"]
selectors = ["mutualinfo"]
metrics = ["ACC", "AUC", "PREC", "RECALL", "F1SCORE", "SPEC", "MCC"]

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

    for selector in selectors:
        print(f" Selector: {selector}")
        selector_path = os.path.join(selector, dataset)
        print(f"  Ruta del selector: {selector_path}")

        # JSON
        data_json = {
            "DATASET_NAME": dataset,
            "SELECTOR": selector,
            "DATA": []
        }

        # Obtener lista de archivos según el selector
        if selector == "mutualinfo":
            print("  Buscando archivos en subcarpetas...")
            split_files = []
            for root, _, files in os.walk(selector_path):
                for f in files:
                    if f.endswith(".csv") and dataset in f:
                        split_files.append(os.path.join(root, f))
            split_files = sorted(split_files)
        else:
            print("  Buscando archivos en carpeta raíz del selector...")
            split_files = sorted([
                os.path.join(selector_path, f)
                for f in os.listdir(selector_path)
                if f.endswith(".csv")
            ])

        print(f"  Archivos encontrados: {len(split_files)}")
        for f in split_files:
            print(f"   - {f}")

        for clf_name, clf in get_classifiers().items():
            print(f"  Clasificador: {clf_name}")
            clf_data = {"CLASSF_NAME": clf_name, "FOLDS": []}

            for split_path in split_files:
                split_file = os.path.basename(split_path)
                print(f"   Procesando archivo de ranking: {split_file}")

                match = re.search(r"fold_?(\d+)", split_file)
                if match:
                    split_id = int(match.group(1))
                    print(f"    Fold detectado: {split_id}")
                else:
                    print(f"    No se pudo extraer el ID del fold de {split_file}")
                    continue

                ranking_df = pd.read_csv(split_path)
                print(f"    Ranking cargado con {len(ranking_df)} features.")

                if selector in ["svmlineal", "randomforest"]:
                    base_name = f"{selector}_train_fold_{split_id}_{dataset}_normalized.csv"
                    base_name2 = f"{selector}_val_fold_{split_id}_{dataset}_normalized.csv"
                elif selector == "mutualinfo":
                    base_name = os.path.basename(split_path)
                    base_name2 = base_name.replace("_train_", "_val_")
                    base_name = re.sub(r"_k\d+", "", base_name)
                    base_name2 = re.sub(r"_k\d+", "", base_name2)
                else:
                    print(f"    Selector desconocido: {selector}")
                    continue

                fold_name = base_name.replace(f"{selector}_", "")
                fold_name2 = base_name2.replace(f"{selector}_", "")
                train_path = f"databases/normalized/{dataset}/folds/train/{fold_name}"
                test_path = f"databases/normalized/{dataset}/folds/val/{fold_name2}"

                print(f"    Train path: {train_path}")
                print(f"    Test path: {test_path}")

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
                print(f"    Columna objetivo detectada: {target_col}")

                # Separamos la variable objetivo (target) del dataframe de entrenamiento y prueba
                y_train = df_train[target_col]
                y_test = df_test[target_col]

                # Preparamos las variables independientes (features)
                X_train_all = df_train.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number])
                X_test_all = df_test.drop(columns=[target_col, "id"], errors='ignore').select_dtypes(include=[np.number])

                split_data = {"SPLIT_ID": split_id, "DATA": []}

                for n in range(1, len(ranking_df) + 1):
                    selected_features = ranking_df["feature"].iloc[:n].values

                    X_train = X_train_all[selected_features]
                    X_test = X_test_all[selected_features]

                    try:
                        # pesos balanceados
                        print("Pesos originales (clases y su cuenta):")
                        unique, counts = np.unique(y_train, return_counts=True)
                        for cls, cnt in zip(unique, counts):
                            print(f"  Clase {cls}: {cnt} muestras")

                        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

                        # Mostrar pesos balanceados por clase
                        unique_classes = np.unique(y_train)
                        class_weights_dict = {
                            cls: (len(y_train) / (len(unique_classes) * count))
                            for cls, count in zip(unique_classes, counts)
                        }

                        print("Pesos balanceados asignados a cada clase:")
                        for cls in sorted(class_weights_dict):
                            print(f"  Clase {cls}: peso {class_weights_dict[cls]:.4f}")

                        # sample_weight solo si el clasificador lo soporta
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
                            "NAME_FEAT": selected_features.tolist(),
                            "DATA": {
                                "TRAIN": metrics_train,
                                "VAL": metrics_val
                            }
                        })
                        if n % 10 == 0 or n == len(ranking_df):
                            print(f"    Progreso: {n}/{len(ranking_df)} features evaluadas.")

                    except Exception as e:
                        print(f"    Error con clasificador {clf_name} en fold {split_id}, n_feats={n}: {e}")
                        continue

                clf_data["FOLDS"].append(split_data)

            data_json["DATA"].append(clf_data)

        os.makedirs("classification_results", exist_ok=True)
        output_path = f"classification_results/{dataset}_{selector}_classification.json"
        with open(output_path, "w") as f:
            json.dump(data_json, f, indent=4)
        print(f"Resultados guardados en: {output_path}")
