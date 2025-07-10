import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Crear carpeta de salida para los splits de validación cruzada
os.makedirs("databases/splits/folds", exist_ok=True)

# Cargar el conjunto de entrenamiento (ya separado del holdout)
df = pd.read_csv("databases/splits/train_gallstone.csv")

# Definir variable target (cambia el nombre si es distinto)
target_column = "Gallstone Status"  # ← Asegúrate de que este es el nombre correcto

# Inicializar StratifiedKFold con 10 divisiones
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Enumerar y guardar los folds
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column]), start=1):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    # Guardar los datasets para este fold
    df_train.to_csv(f"databases/splits/folds/train_fold_{fold}.csv", index=False)
    df_val.to_csv(f"databases/splits/folds/val_fold_{fold}.csv", index=False)

    print(f"Fold {fold}: Train = {len(df_train)} muestras, Val = {len(df_val)} muestras")

print("División en folds completada y guardada en databases/splits/folds/")
