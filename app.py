from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import uuid  # Untuk nama file unik

app = Flask(__name__)

# Load model regresi linear
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Model tidak ditemukan! Pastikan file model.pkl ada.")
    model = None

# Folder untuk menyimpan file dataset dan grafik
UPLOAD_FOLDER = 'dataset/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    graph_url = None
    tables = None
    error = None

    if request.method == "POST":
        if "dataset" in request.files and request.files["dataset"].filename:
            # ➕ Proses unggah CSV
            file = request.files["dataset"]
            if file and file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
                file.save(filepath)

                try:
                    df = pd.read_csv(filepath, sep=",", encoding="utf-8")
                    df.columns = df.columns.str.strip()  # Bersihkan spasi kolom

                    # Kolom yang dibutuhkan
                    required_columns = ['rain_accumulation', 'relative_humidity', 'avg_wind_speed']
                    if not all(col in df.columns for col in required_columns):
                        error = f"Kolom dalam dataset tidak sesuai! Kolom ditemukan: {df.columns.tolist()}"
                        return render_template("index.html", error=error)

                    df = df.fillna(0)  # Tangani missing values

                    df["PrediksiSuhu"] = model.predict(df[required_columns])

                    # Simpan grafik prediksi dari CSV
                    graph_name = f"static/graph_{uuid.uuid4().hex}.png"
                    plt.figure(figsize=(8, 5))
                    plt.plot(df["PrediksiSuhu"], label="Prediksi Suhu", marker='o', linestyle='dashed', color='b')
                    plt.xlabel("Data ke-")
                    plt.ylabel("Suhu (°C)")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(graph_name)
                    plt.close()

                    graph_url = graph_name
                    tables = [df.to_html(classes='table table-bordered')]

                except Exception as e:
                    error = f"Terjadi kesalahan dalam pemrosesan dataset: {e}"

        else:
            # ➕ Proses input manual
            try:
                curah_hujan = float(request.form["curah_hujan"])
                kelembaban = float(request.form["kelembaban"])
                kecepatan_angin = float(request.form["kecepatan_angin"])

                input_data = np.array([[curah_hujan, kelembaban, kecepatan_angin]])
                prediction = model.predict(input_data)[0]

                # Buat grafik prediksi suhu input manual
                graph_name = f"static/graph_{uuid.uuid4().hex}.png"
                plt.figure(figsize=(10, 6))  # Lebar = 10 inch, tinggi = 6 inch
                plt.bar(["Prediksi Suhu"], [prediction], color='#007bff')
                plt.ylabel("Suhu (°C)")
                plt.title("Hasil Prediksi Suhu (Input Manual)")
                plt.tight_layout()
                plt.savefig(graph_name)
                plt.close()

                graph_url = graph_name

            except ValueError:
                error = "Input tidak valid, pastikan angka dimasukkan dengan benar."
            except Exception as e:
                error = f"Terjadi kesalahan: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        graph=graph_url,
        tables=tables,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
