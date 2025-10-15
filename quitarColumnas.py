import pandas as pd


def limpiar_columnas(input_filename):
    input_path = f"databases/processed/{input_filename}"
    output_path = f"databases/processed/{input_filename}"

    df = pd.read_csv(input_path)

    # Filtrar columnas que comienzan con "empty" o "missingindicator" o terminan con "homogeneus"
    columnas_a_eliminar = [col for col in df.columns
                           if col.startswith('empty') or
                           col.startswith('missingindicator') or
                           col.endswith('homogeneous')]

    df_limpio = df.drop(columns=columnas_a_eliminar)

    df_limpio.to_csv(output_path, index=False)
    print(f"Archivo limpio guardado en: {output_path}")


# Ejemplo de uso:
input_filename = 'bacterias2_encoded.csv'
limpiar_columnas(input_filename)
