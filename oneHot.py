import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

input_filename = "toxicity.csv"

input_path = f"databases/original/{input_filename}"
output_filename = input_filename.replace(".csv", "_encoded.csv")
output_path = f"databases/processed/{output_filename}"

# Cargar el dataset
df = pd.read_csv(input_path)

categorical_cols = ['Class']  # Ejemplo: ['Sexo', 'Fumador']

print("Columnas categóricas a codificar:", categorical_cols)

# Inicializar OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Codificar columnas categóricas
encoded_array = encoder.fit_transform(df[categorical_cols])


encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Eliminar columnas originales y combinar con las codificadas
df_numerical = df.drop(columns=categorical_cols)
df_final = pd.concat([df_numerical, encoded_df], axis=1)

print("\nPrimeras filas del DataFrame transformado:")
print(df_final.head())

os.makedirs("databases/processed", exist_ok=True)  # Crear carpeta si no existe
df_final.to_csv(output_path, index=False)
print(f"\n✅ Archivo guardado en: {output_path}")
