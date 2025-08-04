import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler

possible_targets = ["Gallstone Status", "Class", "class", "Label"]

def split_holdout(base_name):
    print(f"\n Split Holdout - Dataset: {base_name}")

    input_filename = f"{base_name}_encoded.csv"
    input_path = f"databases/processed/{input_filename}"
    basename = input_filename.replace("_encoded.csv", "")
    output_train_path = f"databases/splits/train_{basename}.csv"
    output_test_path = f"databases/splits/test_{basename}.csv"

    os.makedirs("databases/splits", exist_ok=True)

    df = pd.read_csv(input_path)

    if 'id'and base_name!="DARWIN" not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))

    df.rename(columns={'ID': 'id'}, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    print(f"Nº muestras total: {len(df)}")
    print(f"Nº muestras entrenamiento: {len(train_df)}")
    print(f"Nº muestras test (holdout): {len(test_df)}")

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    print(f"✅ Conjuntos guardados en:")
    print(f"   - {output_train_path}")
    print(f"   - {output_test_path}")


def cross_validation(base_name):
    print(f"\n Cross-validation Estratificada (10 folds) - Dataset: {base_name}")

    input_train_filename = f"train_{base_name}.csv"
    train_path = f"databases/splits/{input_train_filename}"

    df_train = pd.read_csv(train_path)

    target_column = None
    for col in possible_targets:
        if col in df_train.columns:
            target_column = col
            break

    if target_column is None:
        raise ValueError(f"❌ Ninguna de las columnas objetivo posibles se encontró en {input_train_filename}")

    basename = input_train_filename.replace("train_", "").replace(".csv", "")
    folds_output_dir = f"databases/splits/folds/{basename}"

    os.makedirs(folds_output_dir, exist_ok=True)

    df = pd.read_csv(train_path)

    if target_column not in df.columns:
        raise ValueError(f"❌ La columna '{target_column}' no existe en el archivo {input_train_filename}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
        df_train_fold = df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df.iloc[val_idx].reset_index(drop=True)

        train_fold_path = f"{folds_output_dir}/train_fold_{fold}_{basename}.csv"
        val_fold_path = f"{folds_output_dir}/val_fold_{fold}_{basename}.csv"

        df_train_fold.to_csv(train_fold_path, index=False)
        df_val_fold.to_csv(val_fold_path, index=False)

        print(f"Fold {fold}: Train = {len(df_train_fold)} muestras, Val = {len(df_val_fold)} muestras")

    print(f"\n✅ División en folds completada y guardada en {folds_output_dir}/")


def fit_scale_df(df):
    ids = df['id']
    df_no_id = df.drop(columns=['id'])

    cols_to_exclude = [col for col in df_no_id.columns if col in possible_targets]
    cols_to_scale = df_no_id.drop(columns=cols_to_exclude)

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(cols_to_scale)
    df_scaled = pd.DataFrame(scaled_array, columns=cols_to_scale.columns, index=df.index)

    df_final = pd.concat([ids, df_scaled, df_no_id[cols_to_exclude]], axis=1)
    return df_final, scaler


def transform_df(df, scaler):
    ids = df['id']
    df_no_id = df.drop(columns=['id'])

    cols_to_exclude = [col for col in df_no_id.columns if col in possible_targets]
    cols_to_scale = df_no_id.drop(columns=cols_to_exclude)

    scaled_array = scaler.transform(cols_to_scale)
    df_scaled = pd.DataFrame(scaled_array, columns=cols_to_scale.columns, index=df.index)

    df_final = pd.concat([ids, df_scaled, df_no_id[cols_to_exclude]], axis=1)
    return df_final


def scale_all(base_name):
    print(f"\n Escalado de datos (RobustScaler) - Dataset: {base_name}")

    normalized_base_dir = f"databases/normalized/{base_name}"
    normalized_holdout_dir = f"{normalized_base_dir}/holdout"
    normalized_folds_dir = f"{normalized_base_dir}/folds"

    os.makedirs(normalized_holdout_dir, exist_ok=True)
    os.makedirs(normalized_folds_dir, exist_ok=True)

    train_path = f"databases/splits/train_{base_name}.csv"
    df_train = pd.read_csv(train_path)
    df_train_scaled, global_scaler = fit_scale_df(df_train)
    df_train_scaled.to_csv(f"{normalized_holdout_dir}/train_{base_name}_normalized.csv", index=False)

    print(f"✅ Train escalado guardado en:\n   {normalized_holdout_dir}/")

    test_path = f"databases/splits/test_{base_name}.csv"
    df_test = pd.read_csv(test_path)
    df_test_scaled = transform_df(df_test, global_scaler)
    df_test_scaled.to_csv(f"{normalized_holdout_dir}/test_{base_name}_normalized.csv", index=False)

    print(f"✅ Holdout escalado guardado en:\n   {normalized_holdout_dir}/")

    folds_dir = f"databases/splits/folds/{base_name}"
    for fold in range(1, 11):
        train_fold_path = f"{folds_dir}/train_fold_{fold}_{base_name}.csv"
        val_fold_path = f"{folds_dir}/val_fold_{fold}_{base_name}.csv"

        df_train_fold = pd.read_csv(train_fold_path)
        df_val_fold = pd.read_csv(val_fold_path)

        df_train_fold_scaled, fold_scaler = fit_scale_df(df_train_fold)
        df_val_fold_scaled = transform_df(df_val_fold, fold_scaler)

        df_train_fold_scaled.to_csv(f"{normalized_folds_dir}/train_fold_{fold}_{base_name}_normalized.csv", index=False)
        df_val_fold_scaled.to_csv(f"{normalized_folds_dir}/val_fold_{fold}_{base_name}_normalized.csv", index=False)

        print(f"✅ Fold {fold} escalado y guardado.")

    print("\n Estandarización completa.")


def run_for_dataset(base_name):
    split_holdout(base_name)
    cross_validation(base_name)
    scale_all(base_name)


if __name__ == "__main__":
    datasets = [
        "gallstone",
        "DARWIN",
        "toxicity",
        "DIA_trainingANDTESTset_RDKit_descriptors"
    ]

    for ds in datasets:
        print(f"\n\n=== Procesando dataset: {ds} ===")
        run_for_dataset(ds)
