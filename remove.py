import pandas as pd
import os
import pandas as pd



def remove_non_numeric_and_save(input_filename):
    """
    Lee un CSV desde databases/original/,
    convierte columnas a numéricas cuando sea posible,
    elimina las no numéricas restantes,
    y guarda el resultado en databases/processed/ con sufijo _encoded.csv.
    """
    input_path = f"databases/original/{input_filename}"
    output_filename = input_filename.replace(".csv", "_encoded.csv")
    output_path = f"databases/processed/{output_filename}"

    print(f"\nLeyendo dataset: {input_path}")
    df = pd.read_csv(input_path)

    print(f"📊 Columnas originales: {len(df.columns)}")

    # Intentar convertir todas las columnas a numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Ahora volver a intentar convertir las tipo 'object'
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:  # si hay valores válidos
            df[col] = converted

    # Identificar columnas que siguen sin ser numéricas
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_cols:
        print(f"Eliminando columnas no numéricas: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    else:
        print("No se encontraron columnas no numéricas.")

    # Guardar dataset limpio
    os.makedirs("databases/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nArchivo limpio guardado en: {output_path}")
    print(f"   Columnas finales: {len(df.columns)}")
    print(f"   Nombres de columnas: {list(df.columns)}")


if __name__ == "__main__":
    #remove_non_numeric_and_save("respiratoriovsall.csv")
    df = pd.read_csv("databases/original/respiratoriovsall.csv")

    resultado = (
        df.groupby('ID', as_index=False)['respiratoria']
        .max()
    )
    df = df.drop(columns=['respiratoria'])
    df = df.merge(resultado, on='id', how='left')


    df.to_csv("resultado.csv", index=False)

    print("Archivo  generado correctamente ")