import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Crear carpeta si no existe
os.makedirs("databases/splits", exist_ok=True)

# Cargar el dataset ya procesado previamente
df = pd.read_csv("databases/processed/gallstone_encoded.csv")

# Añadir una columna 'id' si no existe
if 'id' not in df.columns:
    df.insert(0, 'id', range(1, len(df) + 1))

# Dividir en entrenamiento (80%) y test (20%) con holdout
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Mostrar tamaños de los subconjuntos
print(f"Nº muestras total: {len(df)}")
print(f"Nº muestras entrenamiento: {len(train_df)}")
print(f"Nº muestras test (holdout): {len(test_df)}")

# Guardar los subconjuntos
train_df.to_csv("databases/splits/train_gallstone.csv", index=False)
test_df.to_csv("databases/splits/test_gallstone.csv", index=False)

print("Conjuntos guardados en databases/splits/")
