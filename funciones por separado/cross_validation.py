import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

input_train_filename = "train_DARWIN.csv"

possible_targets = ["Gallstone Status", "Class", "class", "Label"]

train_path = f"../databases/splits/{input_train_filename}"
df_train = pd.read_csv(train_path)

# Buscar qué columna existe en df_train entre las posibles
target_column = None
for col in possible_targets:
    if col in df_train.columns:
        target_column = col
        break

# Rutas automáticas
input_path = f"../databases/splits/{input_train_filename}"
basename = input_train_filename.replace("train_", "").replace(".csv", "")

# Nueva ruta para folds dentro de una carpeta con el nombre base
folds_output_dir = f"../databases/splits/folds/{basename}"
os.makedirs(folds_output_dir, exist_ok=True)

# Cargar el conjunto de entrenamiento
df = pd.read_csv(input_path)

# Verificar que la columna target exista
if target_column not in df.columns:
    raise ValueError(f"❌ La columna '{target_column}' no existe en el archivo {input_train_filename}")

# Inicializar StratifiedKFold con 10 divisiones
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Enumerar y guardar los folds
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_fold_path = f"{folds_output_dir}/train_fold_{fold}_{basename}.csv"
    val_fold_path = f"{folds_output_dir}/val_fold_{fold}_{basename}.csv"

    df_train.to_csv(train_fold_path, index=False)
    df_val.to_csv(val_fold_path, index=False)

    print(f"Fold {fold}: Train = {len(df_train)} muestras, Val = {len(df_val)} muestras")

print(f"\n✅ División en folds completada y guardada en {folds_output_dir}/")
