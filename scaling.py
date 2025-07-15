import os
import pandas as pd
from sklearn.preprocessing import robust_scale

base_name = "toxicity"

normalized_base_dir = f"databases/normalized/{base_name}"
normalized_holdout_dir = f"{normalized_base_dir}/holdout"
normalized_folds_dir = f"{normalized_base_dir}/folds"

os.makedirs(normalized_holdout_dir, exist_ok=True)
os.makedirs(normalized_folds_dir, exist_ok=True)

possible_targets = ["Gallstone Status", "Class", "class", "Label"]

def scale_df(df):
    ids = df['id']
    df_no_id = df.drop(columns=['id'])

    # Separar columnas a excluir
    cols_to_exclude = [col for col in df_no_id.columns if col in possible_targets]
    cols_to_scale = df_no_id.drop(columns=cols_to_exclude)

    # Aplicar robust_scale
    scaled_array = robust_scale(cols_to_scale)
    df_scaled = pd.DataFrame(scaled_array, columns=cols_to_scale.columns, index=df.index)

    # Combinar todo
    df_final = pd.concat([ids, df_scaled, df_no_id[cols_to_exclude]], axis=1)
    return df_final

# Escalar holdout (test)
holdout_path = f"databases/splits/test_{base_name}.csv"
df_holdout = pd.read_csv(holdout_path)
df_holdout_scaled = scale_df(df_holdout)
df_holdout_scaled.to_csv(f"{normalized_holdout_dir}/test_{base_name}_normalized.csv", index=False)
print(f"✅ Holdout escalado guardado en {normalized_holdout_dir}/test_{base_name}_normalized.csv")

# Escalar entrenamiento general
train_path = f"databases/splits/train_{base_name}.csv"
df_train = pd.read_csv(train_path)
df_train_scaled = scale_df(df_train)
df_train_scaled.to_csv(f"{normalized_holdout_dir}/train_{base_name}_normalized.csv", index=False)
print(f"✅ Entrenamiento escalado guardado en {normalized_holdout_dir}/train_{base_name}_normalized.csv")

# Escalar folds uno por uno
folds_dir = f"databases/splits/folds/{base_name}"

for fold in range(1, 11):
    train_fold_path = f"{folds_dir}/train_fold_{fold}_{base_name}.csv"
    val_fold_path = f"{folds_dir}/val_fold_{fold}_{base_name}.csv"

    df_train_fold = pd.read_csv(train_fold_path)
    df_val_fold = pd.read_csv(val_fold_path)

    df_train_fold_scaled = scale_df(df_train_fold)
    df_val_fold_scaled = scale_df(df_val_fold)

    df_train_fold_scaled.to_csv(f"{normalized_folds_dir}/train_fold_{fold}_{base_name}_normalized.csv", index=False)
    df_val_fold_scaled.to_csv(f"{normalized_folds_dir}/val_fold_{fold}_{base_name}_normalized.csv", index=False)

    print(f"✅ Fold {fold} escalado y guardado.")

print("\n🎯 Estandarización completa.")
