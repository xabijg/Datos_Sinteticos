import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler

possible_targets = ["HONG","BACT","VIRUS"]
id_column = "id"

def split_holdout(base_name): #Separamos en train y test (80% y 20%)
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



def crossValidation_scall(base_name):
    print(f"\nCrossValidation Estratificada (10 folds) - Dataset: {base_name}")

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

    # Definimos columnas que NO deben ser estandarizadas:
    # - La columna 'id'
    # - La columna objetivo
    exclude_cols = [id_column, target_column]

    # Añadimos valores q estén dentro del rango [0,1]
    for col in df.columns:
        if col in exclude_cols:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if 0.0 <= col_min and col_max <= 1.0:
            exclude_cols.append(col)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Recorremos cada fold, obteniendo índices de train y validación
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
        # Seleccionamos los datos para entrenamiento y validación de este fold
        df_train_fold = df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df.iloc[val_idx].reset_index(drop=True)

        # Objeto
        scaler = RobustScaler()

        # Definimos las columnas que vamos a escalar
        features_to_scale = [col for col in df.columns if col not in exclude_cols]

        # Hacemos copias para no modificar el dataset original
        df_train_fold_scaled = df_train_fold.copy()
        df_val_fold_scaled = df_val_fold.copy()

        # Escalamos SOLO las columnas seleccionadas:
        # - Ajustamos el scaler con los datos de entrenamiento (fit_transform)
        # - Aplicamos la misma transformación al conjunto de validación (transform)
        df_train_fold_scaled[features_to_scale] = scaler.fit_transform(df_train_fold[features_to_scale])
        df_val_fold_scaled[features_to_scale] = scaler.transform(df_val_fold[features_to_scale])

        # Construimos las rutas para guardar los CSVs de cada fold
        train_fold_path = f"{folds_output_dir}/train_fold_{fold}_{basename}.csv"
        val_fold_path = f"{folds_output_dir}/val_fold_{fold}_{basename}.csv"

        df_train_fold_scaled.to_csv(train_fold_path, index=False)
        df_val_fold_scaled.to_csv(val_fold_path, index=False)

        print(f"Fold {fold}: Train = {len(df_train_fold)} muestras, Val = {len(df_val_fold)} muestras")

    print(f"\nDivisión en folds completada y guardada en {folds_output_dir}/")





def run_for_dataset(base_name):
    #split_holdout(base_name)
    crossValidation_scall(base_name)



if __name__ == "__main__":


    datasets = [
        "hongos",
        "bacterias",
        "virus"
    ]

    for ds in datasets:
        print(f"\n\n=== Procesando dataset: {ds} ===")
        run_for_dataset(ds)
