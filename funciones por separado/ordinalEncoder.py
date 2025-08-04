import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os

input_filename = "DARWIN.csv"

input_path = f"../databases/original/{input_filename}"
output_filename = input_filename.replace(".csv", "_encoded.csv")
output_path = f"../databases/processed/{output_filename}"

df = pd.read_csv(input_path)

# Columnas categóricas ordinales
categorical_cols = ['class']

print("Columnas categóricas ordinales a codificar:", categorical_cols)

orden_class = [['H', 'P']]

# Inicializar OrdinalEncoder con el orden definido
encoder = OrdinalEncoder(categories=orden_class)

# Codificar columnas categóricas ordinales
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print("\nPrimeras filas del DataFrame transformado:")
print(df.head())

# Crear carpeta de salida si no existe
os.makedirs("../databases/processed", exist_ok=True)

# Guardar archivo codificado
df.to_csv(output_path, index=False)
print(f"\n✅ Archivo guardado en: {output_path}")
