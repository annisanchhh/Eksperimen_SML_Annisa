import pandas as pd
import os

def preprocess_heart_data(df):
    """
    Preprocessing otomatis dataset Heart Disease
    Output disimpan di folder preprocessing/
    """

    # =========================
    # PATH OUTPUT (PASTI BENAR)
    # =========================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "heart_preprocessing.csv")

    # =========================
    # BINNING AGE
    # =========================
    bins_age = [0, 40, 50, 60, float('inf')]
    df['age_Bin'] = pd.cut(df['age'], bins=bins_age, labels=False)

    # =========================
    # BINNING CHOLESTEROL
    # =========================
    bins_chol = [0, 200, 240, 300, float('inf')]
    df['chol_Bin'] = pd.cut(df['chol'], bins=bins_chol, labels=False)

    # =========================
    # DROP FITUR ASLI
    # =========================
    df = df.drop(columns=['age', 'chol'])

    # =========================
    # SIMPAN CSV
    # =========================
    df.to_csv(output_path, index=False)
    print("✅ SAVED TO:", output_path)

    return df
