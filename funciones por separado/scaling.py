import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

base_name = "toxicity"

# ✅ Rutas corregidas para subcarpeta 'funciones'
normalized_base_dir = f"../databases/normalized/{base_name}"
normalized_holdout_dir = f"{normalized_base_dir}/holdout"
normalized_folds_dir = f"{normalized_base_dir}/folds"

os.makedirs(normalized_holdout_dir, exist_ok=True)
os.makedirs(normalized_folds_dir, exist_ok=True)

possible_targets = ["Gallstone Status", "Class", "class", "Label"]

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

# ✅ Rutas corregidas
train_path = f"../databases/splits/train_{base_name}.csv"
df_train = pd.read_csv(train_path)
df_train_scaled, global_scaler = fit_scale_df(df_train)
df_train_scaled.to_csv(f"{normalized_holdout_dir}/train_{base_name}_normalized.csv", index=False)
print(f"✅ Entrenamiento escalado guardado en {normalized_holdout_dir}/train_{base_name}_normalized.csv")

holdout_path = f"../databases/splits/test_{base_name}.csv"
df_holdout = pd.read_csv(holdout_path)
df_holdout_scaled = transform_df(df_holdout, global_scaler)
df_holdout_scaled.to_csv(f"{normalized_holdout_dir}/test_{base_name}_normalized.csv", index=False)
print(f"✅ Holdout escalado guardado en {normalized_holdout_dir}/test_{base_name}_normalized.csv")

# ✅ Rutas corregidas para folds
folds_dir = f"../databases/splits/folds/{base_name}"

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
