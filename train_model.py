import pandas as pd

# Baca dataset
df = pd.read_csv("dataset/data_cuaca.csv")

# Cek nama kolom agar sesuai dengan yang ada dalam dataset
print("Nama Kolom:", df.columns)

# Pastikan tidak ada data kosong
df = df.dropna()  # Menghapus baris dengan nilai kosong

# Cek tipe data untuk menghindari error dalam model
print("\nTipe Data:\n", df.dtypes)

# Pilih fitur yang digunakan untuk prediksi
try:
    X = df[['rain_accumulation', 'relative_humidity', 'avg_wind_speed']]
    y = df['air_temp']  # Target prediksi
    print("\nContoh Data Input:\n", X.head())
except KeyError as e:
    print("\nError: Kolom yang dipilih tidak ditemukan dalam dataset!", e)
