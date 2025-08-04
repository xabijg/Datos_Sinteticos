import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC

dataset_name = "gallstone"

input_dir = os.path.join("..", "databases", "normalized", dataset_name, "folds")
output_dir = os.path.join("..", "svmlineal", dataset_name)
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.startswith("train_fold") and filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"svmlineal_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        df = pd.read_csv(input_path)
        y = df["Gallstone Status"]
        X = df.drop(columns=["Gallstone Status", "id"])
        X = X.select_dtypes(include=[np.number])

        X_scaled = X

        svm = LinearSVC(penalty='l2', dual=False, max_iter=50000)
        svm.fit(X_scaled, y)

        coefs = np.abs(svm.coef_)
        if coefs.shape[0] > 1:
            feature_importance = np.mean(coefs, axis=0)
        else:
            feature_importance = coefs.flatten()

        ranking = pd.DataFrame({
            "feature": X.columns,
            "importance": feature_importance
        }).sort_values(by="importance", ascending=False)

        ranking.to_csv(output_path, index=False)

        print(f"Ranking guardado en: {output_path}")
