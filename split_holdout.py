import os
import pandas as pd
from sklearn.model_selection import train_test_split


input_filename = "DARWIN_encoded.csv"

# Rutas automáticas
input_path = f"databases/processed/{input_filename}"
basename = input_filename.replace("_encoded.csv", "")
output_train_path = f"databases/splits/train_{basename}.csv"
output_test_path = f"databases/splits/test_{basename}.csv"


os.makedirs("databases/splits", exist_ok=True)


df = pd.read_csv(input_path)

# Añadir columna 'id' si no existe
#if 'id' not in df.columns:
#    df.insert(0, 'id', range(1, len(df) + 1))

df.rename(columns={'ID': 'id'}, inplace=True)

# Dividir en entrenamiento (80%) y test (20%) con holdout
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Mostrar tamaños de los subconjuntos
print(f"Nº muestras total: {len(df)}")
print(f"Nº muestras entrenamiento: {len(train_df)}")
print(f"Nº muestras test (holdout): {len(test_df)}")

# Guardar los subconjuntos
train_df.to_csv(output_train_path, index=False)
test_df.to_csv(output_test_path, index=False)

print(f"\n✅ Conjuntos guardados en:")
print(f"   - {output_train_path}")
print(f"   - {output_test_path}")
