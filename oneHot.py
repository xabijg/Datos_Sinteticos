import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Cargar el dataset
df = pd.read_csv("databases/original/gallstone.csv")

# Reemplaza los nombres dentro de la lista por los que correspondan a tu dataset
categorical_cols = []

# Mostrar las columnas categóricas seleccionadas
print("Columnas categóricas a codificar:", categorical_cols)

# Inicializar OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Codificar columnas categóricas
encoded_array = encoder.fit_transform(df[categorical_cols])

# Crear DataFrame con columnas codificadas
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Eliminar columnas originales
df_numerical = df.drop(columns=categorical_cols)

# Combinar datos numéricos con los codificados
df_final = pd.concat([df_numerical, encoded_df], axis=1)

# Mostrar resultado
print("\nPrimeras filas del DataFrame transformado:")
print(df_final.head())

# Guardar el resultado
df_final.to_csv("databases/processed/gallstone_encoded.csv", index=False)
