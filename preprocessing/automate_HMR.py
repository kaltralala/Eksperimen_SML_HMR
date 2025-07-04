# 1. BLOK IMPOR PUSTAKA
# Mengimpor semua library yang kita butuhkan di awal.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os

# 2. FUNGSI UTAMA PRA-PEMROSESAN
def run_preprocessing(input_path, output_dir):
    """
    Fungsi utama untuk menjalankan seluruh pipeline pra-pemrosesan data.
    Menerjemahkan semua langkah dari notebook ke dalam satu fungsi yang dapat diandalkan.

    Args:
        input_path (str): Path ke file CSV data mentah.
        output_dir (str): Direktori untuk menyimpan hasil pra-pemrosesan.
    """
    print("[START] Memulai pipeline pra-pemrosesan otomatis...")

    # Langkah 1: Muat Data
    # Sama seperti di notebook, langkah pertama adalah memuat data.
    print(f"[1/5] Memuat data dari: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"KESALAHAN: File tidak ditemukan di {input_path}")
        return

    # Langkah 2: Scaling Fitur 'Time' dan 'Amount'
    # Ini adalah implementasi langsung dari insight EDA.
    print("[2/5] Melakukan scaling pada fitur 'Time' dan 'Amount'...")
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Langkah 3: Pemisahan Fitur (X) dan Target (Y)
    # Memisahkan "petunjuk" dari "kunci jawaban".
    print("[3/5] Memisahkan fitur (X) dan target (y)...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Langkah 4: Pembagian Data (Train-Test Split) dengan Stratifikasi
    # Menerapkan pembagian data yang adil dan dapat direproduksi.
    print("[4/5] Membagi data menjadi set latih dan uji dengan stratifikasi...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Langkah 5: Menyimpan Hasil (Artefak)
    # Menyimpan aset data kita yang sudah bersih ke lokasi yang ditentukan.
    print(f"[5/5] Menyimpan hasil ke direktori: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    
    print("[SUCCESS] Pipeline pra-pemrosesan otomatis selesai.")

# 3. BLOK EKSEKUSI UTAMA
if __name__ == "__main__":
    # Bagian ini hanya akan berjalan jika skrip dieksekusi langsung dari terminal.
    # Ini tidak akan berjalan jika skrip diimpor sebagai modul oleh file lain.
    
    # Membuat parser untuk argumen command-line
    parser = argparse.ArgumentParser(description="Script untuk menjalankan pipeline pra-pemrosesan data deteksi fraud.")
    
    # Mendefinisikan argumen yang kita butuhkan: path input dan direktori output.
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="creditcard.csv"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="preprocessing/hasil_preprocessing/"
    )
    
    args = parser.parse_args()

    # Memanggil fungsi utama dengan argumen yang telah di-parse dari terminal.
    run_preprocessing(args.input_path, args.output_dir)