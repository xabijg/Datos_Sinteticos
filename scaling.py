import os
import pandas as pd
from sklearn.preprocessing import robust_scale

# Crear carpetas para guardar resultados
os.makedirs("databases/normalized/holdout", exist_ok=True)
os.makedirs("databases/normalized/folds", exist_ok=True)

def scale_df(df):
    # Guardar id y separar
    ids = df['id']
    df_numeric = df.drop(columns=['id'])

    # Aplicar robust_scale (devuelve numpy array)
    scaled_array = robust_scale(df_numeric)

    # Reconstruir DataFrame escalado con columnas originales
    df_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns, index=df.index)

    # Reinsertar id
    df_scaled.insert(0, 'id', ids)

    return df_scaled

# Escalar holdout (test)
holdout_path = "databases/splits/test_gallstone.csv"
df_holdout = pd.read_csv(holdout_path)
df_holdout_scaled = scale_df(df_holdout)
df_holdout_scaled.to_csv("databases/normalized/holdout/test_gallstone_normalized.csv", index=False)
print(f"Holdout escalado guardado en databases/normalized/holdout/test_gallstone_normalized.csv")

# Escalar entrenamiento general
train_path = "databases/splits/train_gallstone.csv"
df_train = pd.read_csv(train_path)
df_train_scaled = scale_df(df_train)
df_train_scaled.to_csv("databases/normalized/holdout/train_gallstone_normalized.csv", index=False)
print(f"Entrenamiento escalado guardado en databases/normalized/holdout/train_gallstone_normalized.csv")

# Escalar folds uno por uno
folds_dir = "databases/splits/folds"
normalized_folds_dir = "databases/normalized/folds"

for fold in range(1, 11):
    train_fold_path = f"{folds_dir}/train_fold_{fold}.csv"
    val_fold_path = f"{folds_dir}/val_fold_{fold}.csv"

    df_train_fold = pd.read_csv(train_fold_path)
    df_val_fold = pd.read_csv(val_fold_path)

    df_train_fold_scaled = scale_df(df_train_fold)
    df_val_fold_scaled = scale_df(df_val_fold)

    df_train_fold_scaled.to_csv(f"{normalized_folds_dir}/train_fold_{fold}_normalized.csv", index=False)
    df_val_fold_scaled.to_csv(f"{normalized_folds_dir}/val_fold_{fold}_normalized.csv", index=False)

    print(f"Fold {fold} escalado y guardado.")

print("Estandarización completa.")
