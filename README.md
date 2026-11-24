# Klasterisasi Hotspot Kebakaran per Kabupaten/Kota (ADM2) Indonesia

Proyek Akhir Analisis Big Data.

## Deskripsi singkat

Proyek ini mengelompokkan 519 kabupaten/kota (ADM2) di Indonesia berdasarkan karakteristik
hotspot kebakaran dari satelit MODIS (NASA FIRMS, 2002 sampai 2026). Pipeline batch memakai
HDFS untuk penyimpanan, PySpark untuk pemrosesan dan K-Means, GeoPandas untuk pemetaan titik
ke wilayah, scikit-learn untuk DBSCAN dan analisis stabilitas, serta Tableau Public untuk
dashboard.

Pertanyaan utama: tipe wilayah apa saja yang terbentuk dari intensitas, stabilitas
antartahun, dan sifat deteksi hotspot, dan seberapa stabil hasilnya antara data arsip saja
(skenario A) dan arsip + NRT (skenario B)?

Dashboard: (https://public.tableau.com:9443/views/Countofadm2_clusters_min23/KlasterisasiHotspotKebakaranIndonesiaperKabupatenKotaMODIS20022026?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Sumber data dan cara mendapatkannya

| Data | Penyedia | Berkas |
|---|---|---|
| Hotspot MODIS C6.1 (arsip, standard processing) | NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/download/ | `fire_archive_M-C61_820870.csv` |
| Hotspot MODIS C6.1 (near real time) | NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/download/ | `fire_nrt_M-C61_820870.csv` |
| Batas wilayah ADM2 Indonesia | geoBoundaries, https://www.geoboundaries.org | `indonesia_adm2.geojson` |


Cara mendapatkan (unduh manual, tanpa scraping dan tanpa API):
1. Buka halaman FIRMS Download, pilih produk MODIS C6.1 untuk area Indonesia, unduh data
   arsip dan NRT dalam format CSV.

2. Unduh batas ADM2 Indonesia dari geoBoundaries dalam format GeoJSON.
3. Simpan ketiga berkas di folder `data_raw/` dengan nama seperti pada tabel di atas.

Karakteristik dataset:
- Arsip: 1.544.081 baris, 15 kolom, 4 Juli 2002 sampai 31 Juli 2026.
- NRT: 102.494 baris, 14 kolom (tanpa kolom `type`), 1 Agustus sampai 11 Oktober 2026.
- Gabungan: 1.646.575 baris. GeoJSON: 519 fitur poligon.

## Arsitektur pipeline
<img width="2400" height="2062" alt="Pipeline_Proyek_Hotspot" src="https://github.com/user-attachments/assets/0f916446-f677-430c-8421-b6e47fb05e97" />



## Struktur folder

```
code/          skrip pipeline 01 sampai 08
data_raw/      data mentah (tidak diunggah ke GitHub, lihat bagian sumber data)
data_sample/   cuplikan 1.000 baris tiap CSV dan GeoJSON ringan untuk dashboard
models/        label klaster semua metode dan matriks model (masukan clustering)
results/       laporan kualitas data (JSON), tabel hasil (CSV), dan results/dashboard
docs/          laporan, diagram, dan gambar
```

## Kebutuhan

- Ubuntu 24.04 (WSL2), Python 3.12, virtual environment.
- Hadoop (HDFS) dengan Java 8 [isi: versi Hadoop]. Spark membutuhkan Java 17 (dipasang
  berdampingan dan hanya dipakai untuk Spark lewat variabel `JAVA_HOME`).
- Pustaka Python: lihat `requirements.txt` (antara lain pyspark 4.2.1, geopandas 1.2.0,
  scikit-learn 1.9.1, pandas 3.1.0).
- Tableau Public Desktop (Windows) untuk dashboard.

Pasang dependensi:

```bash
python3 -m venv ~/venv-bda
source ~/venv-bda/bin/activate
pip install -r requirements.txt
```

## Langkah menjalankan (berurutan)

Jalankan dari path tanpa spasi (perintah `hdfs dfs -put` tidak membaca path ber-spasi).

```bash
export PROJ=/path/ke/repositori
export HDFS_URI=hdfs://localhost:9000
export JAVA17=/usr/lib/jvm/java-17-openjdk-amd64

# 0. Nyalakan HDFS dan unggah data mentah
start-dfs.sh
hdfs dfs -mkdir -p /proyek_akhir/raw /proyek_akhir/clean /proyek_akhir/features /proyek_akhir/results
hdfs dfs -put -f $PROJ/data_raw/fire_archive_M-C61_820870.csv /proyek_akhir/raw/
hdfs dfs -put -f $PROJ/data_raw/fire_nrt_M-C61_820870.csv /proyek_akhir/raw/
hdfs dfs -put -f $PROJ/data_raw/indonesia_adm2.geojson /proyek_akhir/raw/

# 1-4. Pemrosesan dan K-Means (Spark, Java 17)
JAVA_HOME=$JAVA17 python $PROJ/code/01_ingest_quality.py
JAVA_HOME=$JAVA17 python $PROJ/code/02_spatial_mapping.py
JAVA_HOME=$JAVA17 python $PROJ/code/03_build_features.py
JAVA_HOME=$JAVA17 python $PROJ/code/04_kmeans.py

# 5-6. Stabilitas, DBSCAN, Hybrid, Ensemble (tanpa Spark)
python $PROJ/code/05_dbscan_stability.py
python $PROJ/code/06_hybrid_ensemble.py

# 7. Porsi hotspot bukan vegetasi (Spark), lalu 8. ekspor dashboard (tanpa Spark)
JAVA_HOME=$JAVA17 python $PROJ/code/07_type_profile.py
python $PROJ/code/08_dashboard_export.py
```

Ambang wilayah dapat diubah, misalnya `MIN_TOTAL=10` pada skrip 04 sampai 08.

| Skrip | Fungsi | Keluaran utama |
|---|---|---|
| 01_ingest_quality.py | Gabung arsip + NRT, pemeriksaan kualitas | `clean/hotspot_merged`, `quality_report_stage1.json` |
| 02_spatial_mapping.py | Pemetaan titik ke ADM2 (GeoPandas) | `clean/hotspot_adm2`, `clean/coord_adm2_map` |
| 03_build_features.py | Fitur wilayah skenario A dan B | `features/region_features_A`, `_B` |
| 04_kmeans.py | Matriks model dan K-Means k=2 sampai 10 | `features/model_matrix_*`, `results/kmeans_*` |
| 05_dbscan_stability.py | Stabilitas 20 seed dan grid DBSCAN | `results/dbscan_grid_*`, `docs/figures` |
| 06_hybrid_ensemble.py | Hybrid, Ensemble, perbandingan metode | `results/clusters_all_methods_min23.csv` |
| 07_type_profile.py | Porsi hotspot per jenis sumber | `results/region_type_share.csv` |
| 08_dashboard_export.py | Data untuk Tableau Public | `results/dashboard/*` |

## Parameter penting

Ambang wilayah total hotspot minimal 23 (461 wilayah), transformasi `log1p` pada
`hotspot_per_bulan` dan `rata2_frp` lalu z-score, seed 42, DBSCAN `eps` 0,9 dan
`min_samples` 5, K-Means k=4 sebagai acuan utama.

## Hasil utama

- 461 dari 519 wilayah ikut clustering. K-Means k=4 (skenario A): 193 wilayah intensitas
  sedang dan stabil, 102 intensitas tinggi, 90 intensitas rendah tetapi tidak stabil, dan
  76 dominan sumber non-vegetasi.
- DBSCAN menunjukkan data wilayah membentuk satu massa padat, sehingga lebih berguna untuk
  mendeteksi wilayah ekstrem (79 wilayah noise pada skenario A).
- Detail lengkap ada di folder `results/` dan pada laporan.

## Keterbatasan

- Kolom `type` hanya ada pada data arsip, dan sebagian hotspot berasal dari sumber
  non-kebakaran (gunung api, sumber statis darat, lepas pantai).
- Data NRT jatuh pada musim puncak (Agustus sampai Oktober 2026), sehingga skenario B
  cenderung menaikkan intensitas.
- Stabilitas antarseed diuji dengan scikit-learn, sedangkan model utama memakai Spark MLlib.
