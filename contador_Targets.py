import os
import pandas as pd
import csv

folder_path = "databases/normalized"
results_folder = os.path.join(folder_path, "results")
os.makedirs(results_folder, exist_ok=True)  # Crear carpeta results si no existe

possible_targets = ["Gallstone Status", "Class", "class", "Label"]

# Descripción para la columna Gallstone Status
descriptions = {
    "Gallstone Status": "Target variable, Gallstones present(1), and absent(0)"
}

def count_targets_in_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print(f"Advertencia: Archivo vacío o sin datos válidos, se ignora: {filepath}")
        return []
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return []

    counts = []
    for target_col in possible_targets:
        if target_col in df.columns:
            value_counts = df[target_col].value_counts(dropna=False).to_dict()
            for value, count in value_counts.items():
                counts.append({
                    "archivo": filepath,
                    "columna_target": target_col,
                    "valor": value,
                    "cantidad": count,
                    "descripcion": descriptions.get(target_col, "")
                })
    return counts

def main():
    results_path = os.path.join(results_folder, "results.csv")
    all_counts = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                counts = count_targets_in_file(filepath)
                all_counts.extend(counts)

    # Guardar resultados por archivo
    with open(results_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["archivo", "columna_target", "valor", "cantidad", "descripcion"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_counts:
            writer.writerow(row)

    print(f"Resultados guardados en: {results_path}")

    # Ahora sumatorios totales por columna target y valor, agregando la descripción para Gallstone Status
    df_all = pd.DataFrame(all_counts)
    if df_all.empty:
        print("No se encontraron datos para resumir.")
        return

    print("\n=== Sumatorios totales por columna target y valor ===")
    summary = df_all.groupby(["columna_target", "valor", "descripcion"])["cantidad"].sum().reset_index()
    print(summary.to_string(index=False))

    # Opcional: guardar resumen en otro archivo CSV
    summary_path = os.path.join(results_folder, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Resumen guardado en: {summary_path}")

if __name__ == "__main__":
    main()
