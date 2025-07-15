import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os

input_filename = "DARWIN.csv"

input_path = f"databases/original/{input_filename}"
output_filename = input_filename.replace(".csv", "_encoded.csv")
output_path = f"databases/processed/{output_filename}"

# Cargar el dataset
df = pd.read_csv(input_path)

#categorical_cols = ['Class']
categorical_cols = ['class']

print("Columnas categóricas ordinales a codificar:", categorical_cols)

# Definir el orden explícito para la columna 'Class'
#orden_class = [['NonToxic', 'Toxic']]
orden_class = [['H', 'P']]

# Inicializar OrdinalEncoder con el orden definido
encoder = OrdinalEncoder(categories=orden_class)

# Codificar columnas categóricas ordinales
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print("\nPrimeras filas del DataFrame transformado:")
print(df.head())

os.makedirs("databases/processed", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n✅ Archivo guardado en: {output_path}")
