import pandas as pd
import os

def preprocess_heart_data(df):
    """
    Preprocessing otomatis dataset Heart Disease
    dan menyimpan hasil ke preprocessing/preprocessing_heart/
    """

    # =========================
    # PATH OUTPUT
    # =========================
    output_dir = "preprocessed"
    output_file = "heart_preprocessed.csv"
    output_path = os.path.join(output_dir, output_file)

    # Pastikan folder ada
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # BINNING AGE
    # =========================
    bins_age = [0, 40, 50, 60, float('inf')]
    df['age_Bin'] = pd.cut(
        df['age'],
        bins=bins_age,
        labels=False
    )

    # =========================
    # BINNING CHOLESTEROL
    # =========================
    bins_chol = [0, 200, 240, 300, float('inf')]
    df['chol_Bin'] = pd.cut(
        df['chol'],
        bins=bins_chol,
        labels=False
    )

    # =========================
    # DROP FITUR ASLI
    # =========================
    df = df.drop(columns=['age', 'chol'])

    # =========================
    # SIMPAN KE CSV
    # =========================
    df.to_csv(output_path, index=False)

    return df
