
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler


possible_targets = ["respiratoria"]
id_column = "id"

# En esta funcion no hay tutia, tiene que estar bien
def split_holdout(base_name): # Separamos en train y test (80% y 20%)
    print(f"\n Split Holdout - Dataset: {base_name}")

    input_filename = f"{base_name}_encoded.csv"
    input_path = f"databases/processed/{input_filename}"
    basename = input_filename.replace("_encoded.csv", "")
    output_train_path = f"databases/splits/train_{basename}.csv"
    output_test_path = f"databases/splits/test_{basename}.csv"

    os.makedirs("databases/splits", exist_ok=True)

    df = pd.read_csv(input_path)

    if 'id' not in df.columns and base_name != "DARWIN":
        df.insert(0, 'id', range(1, len(df) + 1))

    df.rename(columns={'ID': 'id'}, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    print(f"Nº muestras total: {len(df)}")
    print(f"Nº muestras entrenamiento: {len(train_df)}")
    print(f"Nº muestras test (holdout): {len(test_df)}")

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    print(f" Conjuntos guardados en:")
    print(f"   - {output_train_path}")
    print(f"   - {output_test_path}")


# Cross-Validation y escalado post seleccion
# def crossValidation_scall(base_name):
#     print(f"\nCrossValidation Estratificada (5 folds) - Dataset: {base_name}")
#
#     input_train_filename = f"train_{base_name}.csv"
#     train_path = f"databases/splits/{input_train_filename}"
#     df = pd.read_csv(train_path)
#
#     target_column = None
#     for col in possible_targets:
#         if col in df.columns:
#             target_column = col
#             break
#
#     if target_column is None:
#         raise ValueError(f"Ninguna de las columnas objetivo posibles se encontró en {input_train_filename}")
#
#     basename = input_train_filename.replace("train_", "").replace(".csv", "")
#     folds_output_dir = f"databases/splits/folds/{basename}"
#     os.makedirs(folds_output_dir, exist_ok=True)
#
#     exclude_cols = [id_column, target_column]
#
#     # excluir columnas en [0,1]
#     for col in df.columns:
#         if col in exclude_cols:
#             continue
#         col_min = df[col].min()
#         col_max = df[col].max()
#         if 0.0 <= col_min and col_max <= 1.0:
#             exclude_cols.append(col)
#
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
#         df_train_fold = df.iloc[train_idx].reset_index(drop=True)
#         df_val_fold = df.iloc[val_idx].reset_index(drop=True)
#
#         # WRAPPER: Selección de características con RFECV
#         X_train = df_train_fold.drop(columns=exclude_cols)
#         y_train = df_train_fold[target_column]
#
#         # Guardamos índice original de cada feature antes de seleccionar
#         feature_indices = {feature: idx for idx, feature in enumerate(X_train.columns)}
#
#         estimator = ExtraTreesClassifier(random_state=42, n_jobs=-1)
#
#         selector = RFECV(
#             estimator=estimator,
#             step=1,
#             cv=5,
#             scoring='f1',
#             n_jobs=-1,
#             min_features_to_select=25
#         )
#
#         selector.fit(X_train, y_train)
#         selected_features = X_train.columns[selector.support_].tolist()  # Lista con las columnas seleccionadas
#
#         print(f"Fold {fold}: {len(selected_features)} características seleccionadas")
#
#         # Entrenamos el modelo final sobre las características seleccionadas (sin aplicar selector adicional)
#         X_selected = X_train[selected_features]
#         model_final = ExtraTreesClassifier(n_estimators=150,random_state=42, n_jobs=-1)
#         model_final.fit(X_selected, y_train)
#
#         importances = model_final.feature_importances_
#
#         # Creamos el ranking con índice original de las features
#         ranking_df = pd.DataFrame({
#             "index": [feature_indices[f] for f in selected_features],
#             "feature": selected_features,
#             "importance": importances
#         }).sort_values(by="importance", ascending=False)
#
#         # Guardamos el ranking
#         importances_dir = f"selector/sinselector_importances/{basename}"
#         os.makedirs(importances_dir, exist_ok=True)
#         ranking_path = f"{importances_dir}/importance_fold_{fold}.csv"
#         ranking_df.to_csv(ranking_path, index=False)
#         print(f" Importancias guardadas en: {ranking_path}")
#
#         # Reducimos los conjuntos a columnas seleccionadas
#         df_train_fold = df_train_fold[exclude_cols + selected_features]
#         df_val_fold = df_val_fold[exclude_cols + selected_features]
#
#         # ESCALADO después de selección
#         scaler = RobustScaler()
#         df_train_fold_scaled = df_train_fold.copy()
#         df_val_fold_scaled = df_val_fold.copy()
#
#         df_train_fold_scaled[selected_features] = scaler.fit_transform(df_train_fold[selected_features])
#         df_val_fold_scaled[selected_features] = scaler.transform(df_val_fold[selected_features])
#
#         df_train_fold_scaled.to_csv(f"{folds_output_dir}/train_fold_{fold}_{basename}.csv", index=False)
#         df_val_fold_scaled.to_csv(f"{folds_output_dir}/val_fold_{fold}_{basename}.csv", index=False)
#
#         print(f"Fold {fold}: Train = {len(df_train_fold)} muestras, Val = {len(df_val_fold)} muestras")
#
#     print(f"\nDivisión en folds completada y guardada en {folds_output_dir}/")



# Cross-Validation y escalado pre seleccion
def crossValidation_scall(base_name):
    print(f"\nCrossValidation Estratificada (5 folds) - Dataset: {base_name}")

    input_train_filename = f"train_{base_name}.csv"
    train_path = f"databases/splits/{input_train_filename}"
    df = pd.read_csv(train_path)

    target_column = None
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break

    if target_column is None:
        raise ValueError(f"Ninguna de las columnas objetivo posibles se encontró en {input_train_filename}")

    basename = input_train_filename.replace("train_", "").replace(".csv", "")
    folds_output_dir = f"databases/splits/folds/{basename}"
    os.makedirs(folds_output_dir, exist_ok=True)

    exclude_cols = [id_column, target_column]

    # excluir columnas en [0,1]
    for col in df.columns:
        if col in exclude_cols:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if 0.0 <= col_min and col_max <= 1.0:
            exclude_cols.append(col)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
        df_train_fold = df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df.iloc[val_idx].reset_index(drop=True)

        # Definir X e y para train y validación (sin excluir todavía)
        X_train_full = df_train_fold.drop(columns=[target_column])
        y_train = df_train_fold[target_column]

        X_val_full = df_val_fold.drop(columns=[target_column])
        y_val = df_val_fold[target_column]

        # Definir columnas para escalar (todas menos exclude_cols)
        features_to_scale = [col for col in X_train_full.columns if col not in exclude_cols]

        scaler = RobustScaler()
        X_train_scaled = X_train_full.copy()
        X_val_scaled = X_val_full.copy()

        # Escalamos solo las features numéricas (excluyendo id y target)
        X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_full[features_to_scale])
        X_val_scaled[features_to_scale] = scaler.transform(X_val_full[features_to_scale])

        # Ahora hacemos selección de características sobre el conjunto escalado
        X_train = X_train_scaled.drop(columns=exclude_cols, errors='ignore')

        # Guardamos índice original de cada feature antes de seleccionar
        feature_indices = {feature: idx for idx, feature in enumerate(X_train.columns)}

        estimator = ExtraTreesClassifier(random_state=42, n_jobs=-1)

        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            min_features_to_select=25
        )

        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_].tolist()  # Lista con las columnas seleccionadas

        print(f"Fold {fold}: {len(selected_features)} características seleccionadas")

        X_selected = X_train[selected_features]
        model_final = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model_final.fit(X_selected, y_train)

        importances = model_final.feature_importances_
        #importances = selector.estimator_.feature_importances_ #PROBAR ESTO!!!!!

        # Creamos el ranking con índice original de las features
        ranking_df = pd.DataFrame({
            "index": [feature_indices[f] for f in selected_features],
            "feature": selected_features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Guardamos el ranking
        importances_dir = f"selector/sinselector_importances/{basename}"
        os.makedirs(importances_dir, exist_ok=True)
        ranking_path = f"{importances_dir}/importance_fold_{fold}.csv"
        ranking_df.to_csv(ranking_path, index=False)
        print(f" Importancias guardadas en: {ranking_path}")

        # Reducimos los conjuntos a columnas seleccionadas + exclude_cols (como id y target)
        cols_final = exclude_cols + selected_features

        df_train_fold_scaled = pd.concat([
            df_train_fold[exclude_cols],
            X_train_scaled[selected_features]
        ], axis=1)

        df_val_fold_scaled = pd.concat([
            df_val_fold[exclude_cols],
            X_val_scaled[selected_features]
        ], axis=1)

        df_train_fold_scaled.to_csv(f"{folds_output_dir}/train_fold_{fold}_{basename}.csv", index=False)
        df_val_fold_scaled.to_csv(f"{folds_output_dir}/val_fold_{fold}_{basename}.csv", index=False)

        print(f"Fold {fold}: Train = {len(df_train_fold_scaled)} muestras, Val = {len(df_val_fold_scaled)} muestras")

    print(f"\nDivisión en folds completada y guardada en {folds_output_dir}/")


def run_for_dataset(base_name):
    split_holdout(base_name)
    crossValidation_scall(base_name)



if __name__ == "__main__":
    datasets = [
        "respiratoriovsall"
    ]

    for ds in datasets:
        print(f"\n\n=== Procesando dataset: {ds} ===")
        run_for_dataset(ds)
