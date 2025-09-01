import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def encode_with_ordinal(input_filename):
    input_path = f"databases/original/{input_filename}"
    output_filename = input_filename.replace(".csv", "_encoded.csv")
    output_path = f"databases/processed/{output_filename}"

    df = pd.read_csv(input_path)

    # Definir columnas categóricas y orden específico según dataset
    if input_filename.lower() == "darwin.csv":
        categorical_cols = ['class']
        categories = [['H', 'P']]  # ejemplo orden para DARWIN
    elif input_filename.lower() == "toxicity.csv":
        categorical_cols = ['Class']
        categories = [['NonToxic', 'Toxic']]  # ejemplo orden para toxicity
    else:
        raise ValueError("Dataset no válido")

    print(f"Columnas categóricas ordinales a codificar en {input_filename}:", categorical_cols)
    encoder = OrdinalEncoder(categories=categories)

    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    print("\nPrimeras filas del DataFrame transformado:")
    print(df.head())

    os.makedirs("databases/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Archivo guardado en: {output_path}")

def encode_with_onehot(input_filename):
    input_path = f"databases/original/{input_filename}"
    output_filename = input_filename.replace(".csv", "_encoded.csv")
    output_path = f"databases/processed/{output_filename}"

    df = pd.read_csv(input_path)

    # Aquí defines las columnas categóricas para OneHotEncoder (ajusta según dataset)
    categorical_cols = ['Class']  # ejemplo

    print(f"Columnas categóricas a codificar con OneHot en {input_filename}:", categorical_cols)

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df_numerical = df.drop(columns=categorical_cols)
    df_final = pd.concat([df_numerical, encoded_df], axis=1)

    print("\nPrimeras filas del DataFrame transformado:")
    print(df_final.head())

    os.makedirs("databases/processed", exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Archivo guardado en: {output_path}")


def wine(input_filename):
    input_path = f"databases/original/{input_filename}"
    output_filename = input_filename.replace(".csv", "_encoded.csv")
    output_path = f"databases/processed/{output_filename}"

    df = pd.read_csv(input_path)

    # Si el dataset es el esperado
    if input_filename.lower() == "winequality-red.csv":
        # Convertir 'quality' a 0 si <=5, y a 1 si >5
        df['class'] = (df['class'] > 5).astype(int)
    else:
        raise ValueError("Dataset no válido")

    print("\nPrimeras filas del DataFrame transformado:")
    print(df.head())

    os.makedirs("databases/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Archivo guardado en: {output_path}")


def main(input_filename):
    # Normalizar nombre para evitar case issues
    filename_lower = input_filename.lower()

    if filename_lower in ["darwin.csv", "toxicity.csv"]:
        encode_with_ordinal(input_filename)
    elif filename_lower in ["winequality-red.csv"]:
        wine(input_filename)
    else:
        encode_with_onehot(input_filename)


# Ejemplo de uso:
if __name__ == "__main__":
    # main("toxicity.csv")
    main("winequality-red.csv")
    main("diabetes_encoded.csv")
    main("toxicity.csv")
    # main("DARWIN.csv")
    # main("DIA_trainingANDTESTset_RDKit_descriptors.csv")
    # main("gallstone.csv")
    # Habria que poner en OneHot el target correspondiente pero no se esta haciendo nada porque ya esta bien pero hay que tener en processed los archivos con sus nombres