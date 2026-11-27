# G6 Advanced AI Model

Proyek klasifikasi status mahasiswa menggunakan Decision Tree, Random Forest,
K-Nearest Neighbors (KNN), Logistic Regression, Artificial Neural Network (ANN),
dan ensemble learning. Alur pengerjaan dimulai dari preprocessing data,
pelatihan dan evaluasi model, hingga pembuatan CSV prediksi.

## 1. Preparation

- Siapkan Anaconda atau Miniconda untuk Virtual Envoronment
- Pakai Python 3.12 di environment Conda agar sesuai dengan metadata notebook
- Buka terminal di folder utama proyek (`G6_AdvancedAI_Model`)

Struktur folder utama:

```text
G6_AdvancedAI_Model/
├── README.md
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── cleaned/
│   ├── preprocessing.ipynb
│   ├── DT_RF.ipynb
│   ├── knn.ipynb
│   ├── logistic-regression.ipynb
│   ├── artificial_neural_network.ipynb
│   ├── ensemble_learning.ipynb
│   └── models.py
└── output/
    ├── *.pkl
    └── csv/
```

## 2. Membuat environment Conda

Jalankan dari folder utama proyek melalui Anaconda Prompt atau terminal yang
sudah mendukung Conda:

```powershell
conda create --name advanced_model python=3.12
conda activate advanced_model
```

Jika environment `advanced_model` sudah ada, cukup jalankan
`conda activate advanced_model` tanpa membuatnya ulang.

Setelah aktivasi, prompt biasanya menampilkan `(advanced_model)`. Periksa versi
Python dan pastikan interpreter berasal dari environment tersebut:

```powershell
python --version
```

## 3. Menginstal library

Dengan environment `advanced_model` aktif, jalankan:

```powershell
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn joblib jupyterlab ipykernel
```

Daftarkan kernel agar environment mudah dipilih di Jupyter/VS Code:

```powershell
python -m ipykernel install --user --name advanced_model --display-name "advanced_model"
```

## 4. Menjalankan notebook

Pastikan `train.csv`, `test.csv`, dan `sample_submission.csv` tersedia di
`src/data/`. Dari folder utama proyek, siapkan direktori hasil dan buka Jupyter:

```powershell
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ('src/data/cleaned', 'output/csv')]"
cd src
python -m jupyterlab
```

Di JupyterLab, buka notebook, pilih kernel **advanced_model**, lalu
jalankan seluruh sel berurutan dari atas ke bawah dengan **Run All Cells**.
Jika kernel belum muncul, pastikan langkah pendaftaran kernel sudah dijalankan
dari environment `advanced_model`, lalu buka ulang JupyterLab.

Gunakan urutan berikut:

| Urutan | Notebook | Fungsi |
| --- | --- | --- |
| 1 | `preprocessing.ipynb` | Menyiapkan data latih, validasi, dan test di `data/cleaned/`. |
| 2 | `DT_RF.ipynb` | Melatih dan mengevaluasi Decision Tree serta Random Forest. |
| 3 | `knn.ipynb` | Melatih dan mengevaluasi KNN. |
| 4 | `logistic-regression.ipynb` | Melatih dan mengevaluasi Logistic Regression. |
| 5 | `artificial_neural_network.ipynb` | Melatih dan mengevaluasi ANN. |
| 6 | `ensemble_learning.ipynb` | Memuat model dasar dan menjalankan ensemble. |

Notebook urutan 2–5 dapat dijalankan dalam urutan bebas setelah preprocessing.
Selesaikan semuanya sebelum menjalankan ensemble agar file model tersedia.
`models.py` berisi definisi kelas yang diimpor oleh ensemble; file ini bukan
program utama yang perlu dijalankan sendiri.

```powershell
conda deactivate
```

## 5. Hasil eksekusi

- Data hasil preprocessing: `src/data/cleaned/`.
- Model dan konfigurasi yang disimpan: `output/*.pkl`.
- CSV prediksi: `output/csv/`, misalnya `submission_random_forest.csv`,
  `submission_knn_euclidean.csv`, `submission_logres_diy.csv`,
  `submission_ann_v3.csv`, dan `submission_ensemble_weighted.csv`.
- Evaluasi dan grafik: ditampilkan pada sel output masing-masing notebook.

## 6. Pembagian Tugas Anggota Kelompok

| Nama | NIM | Tugas |
| --- | --- | --- |
| Umar Hamzah Ramadhan | 23526009 | KNN, Logistic Regression, dan Ensemble Models. |
| Muhammad Fida Raditya | 23526011 | Decision Tree dan Random Forest. |
| Muhammad Fadhlan Ashila Harashta | 23526018 | Artificial Neural Network. |
