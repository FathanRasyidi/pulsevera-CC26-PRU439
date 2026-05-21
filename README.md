# Pulsevera — Predict, Prevent, Prevail
**Coding Camp 2026 powered by DBS Foundation | CC26-PRU439**

> Sistem prediksi risiko penyakit jantung berbasis gaya hidup menggunakan machine learning, diakses melalui aplikasi web.

---

## Tim

| Nama | Role | Status |
|---|---|---|
| Muh. Tegar Adyaksa | Data Scientist | Aktif |
| Afifah Nurazizah | Data Scientist | Tidak Aktif |
| Fathan Rasyidi Mustafa | AI Engineer | Aktif |
| Shafira Kurnia Fasya | AI Engineer | Aktif |
| Muhammad Rifqi Indria Nugraha | Full-Stack Web Developer | Aktif |
| Khoerunnisa | Full-Stack Web Developer | Tidak Aktif |

---

## Dataset

Dataset **tidak disertakan di repository ini** karena ukurannya besar (>100MB).

**Download dataset di sini:**
> 📁 [Google Drive - Pulsevera Data](https://drive.google.com/drive/folders/1jtkudb-Ggt4nk9gZygS4O87hWGDT0sbH?usp=sharing)

Setelah download, letakkan file sesuai struktur berikut:
```
pulsevera-cc26-pru439/
└── data/
    ├── raw/dataset_raw.csv
    ├── processed/dataset_cleaned.csv
    └── final/
        ├── X_train.csv        (356.105 baris × 46 fitur)
        ├── X_test.csv         (89.027 baris × 46 fitur)
        ├── y_train.csv
        ├── y_test.csv
        └── dataset_final.csv
```

---

## Struktur Repository

```
pulsevera-cc26-pru439/
├── notebooks/
│   ├── 01_data_wrangling.ipynb       ← Data Gathering, Assessing, Cleaning
│   ├── 02_eda.ipynb                  ← EDA + 5 Business Questions
│   ├── 03_feature_engineering.ipynb  ← 6 Fitur Baru + Train-Test Split
│   ├── 04_ab_testing.ipynb           ← 4 Hipotesis A/B Testing
│   ├── 05_ml_baseline.ipynb          ← Training 3 Model ML + SMOTE + Threshold Tuning
│   ├── 06_deep_learning.ipynb        ← Deep Learning (Functional API + Focal Loss)
│   ├── 07_evaluation.ipynb           ← Evaluasi Final ML vs DL
│   └── figures/                      ← Visualisasi PNG
├── dashboard/
│   └── app.py                        ← Streamlit Dashboard (Data Science)
├── ml-api/
│   ├── main.py                       ← FastAPI inference endpoint
│   ├── inference.py                  ← Inference module (load model + preprocessing)
│   ├── requirements.txt
│   └── models/
│       ├── pulsevera_dl_model.keras  ← Model Deep Learning (TensorFlow)
│       ├── pulsevera_ml_model.pkl    ← Model ML Baseline (Logistic Regression)
│       ├── scaler.pkl                ← StandardScaler
│       ├── feature_order.pkl         ← Urutan kolom fitur
│       └── threshold.pkl             ← Threshold prediksi (0.3)
├── backend/                          ← [Full-Stack] Node.js/Express
├── frontend/                         ← [Full-Stack] React App
└── data_dictionary.md                ← Dokumentasi lengkap 47 kolom
```

---

## Quick Start

### Streamlit Dashboard
```bash
pip install streamlit plotly pandas scikit-learn scipy
streamlit run dashboard/app.py
```

### Jupyter Notebooks
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly imbalanced-learn joblib tensorflow
jupyter notebook notebooks/
```

### ML API (FastAPI)
```bash
cd ml-api
pip install -r requirements.txt
uvicorn main:app --reload
```
Buka `http://localhost:8000/docs` untuk dokumentasi interaktif.

### Inference Langsung (CLI)
```bash
cd ml-api
python inference.py
```

---

## Progress

### Data Science ✅ Selesai
- [x] Data Gathering & Wrangling (`01_data_wrangling.ipynb`)
- [x] EDA — 5 pertanyaan bisnis, 13 visualisasi (`02_eda.ipynb`)
- [x] Feature Engineering — 6 fitur baru (`03_feature_engineering.ipynb`)
- [x] A/B Testing — 4 hipotesis (`04_ab_testing.ipynb`)
- [x] Dashboard Streamlit (`dashboard/app.py`)
- [x] Data siap untuk model (`data/final/`)
- [x] Data Dictionary (`data_dictionary.md`)
- [ ] Laporan Teknis PDF

### AI Engineer 🔶 Sebagian
- [x] Training 3 model ML baseline — LR, DT, RF + SMOTE (`05_ml_baseline.ipynb`)
- [x] Threshold tuning — Recall ≥ 70%, Accuracy ≥ 85%
- [x] Deep Learning — TensorFlow Functional API + Focal Loss (`06_deep_learning.ipynb`)
- [x] Evaluasi final ML vs DL — model DL dipilih (`07_evaluation.ipynb`)
- [x] Model disimpan format produksi (`.keras`, `.pkl`)
- [x] Inference module (`ml-api/inference.py`)
- [x] FastAPI endpoint `/api/v1/predict` aktif (`ml-api/main.py`)
##### Side-quest ---------------
- [ ] SHAP untuk interpretabilitas prediksi
- [ ] tf.gradient custom loop
- [ ] Generative AI fitur sekunder (on going)
- [x] Tensorboard

### Full-Stack 🔄 In Progress
- [ ] Setup GitHub repo & mockup UI/UX (Figma)
- [ ] Frontend React — form input 10 field
- [ ] Backend Node.js/Express — proxy ke ML API
- [ ] Integrasi frontend ↔ backend ↔ ML API
- [ ] Deployment (Netlify/Vercel + Railway/Render)

---

## API Endpoint

### POST `/api/v1/predict`

**Request:**
```json
{
  "sex": "Male",
  "age_category": 9,
  "height_meters": 1.70,
  "weight_kg": 85.0,
  "sleep_hours": 5.0,
  "physical_activities": "No",
  "smoker_status": "Current-every",
  "alcohol": "Yes",
  "general_health": "Fair",
  "diabetes": "No"
}
```

**Response:**
```json
{
  "risk_score": 0.73,
  "risk_label": "Tinggi",
  "top_risk_factors": ["IsActiveSmoker", "IsObese", "AgeCategory"],
  "recommendations": [
    "Berhenti merokok — konsultasikan program berhenti merokok ke dokter.",
    "Turunkan berat badan dengan diet seimbang dan olahraga rutin.",
    "Lakukan pemeriksaan jantung rutin setiap tahun."
  ]
}
```

### GET `/health`
```json
{"status": "ok", "model_loaded": true}
```

---

## Hasil Evaluasi Model

| Metrik | ML Model (LR) | DL Model | Target |
|---|---|---|---|
| Accuracy | 87.1% | 86.0% | ≥ 85% |
| Recall | 70.2% | 72.0% | ≥ 70% |
| ROC-AUC | 0.882 | 0.884 | ≥ 0.80 |
| Threshold | 0.3 | 0.3 | — |

Model DL dipilih sebagai model utama API karena Recall dan ROC-AUC lebih tinggi.

---

## Arsitektur Sistem

```
[User Browser]
      │
      ▼
┌─────────────────────┐
│   React Frontend    │  Vite · Tailwind CSS · Axios · Recharts
│   (Netlify/Vercel)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Node.js/Express    │  Backend Full-Stack
│  Backend            │  (Railway / Render)
└──────────┬──────────┘
           │ POST /api/v1/predict
           ▼
┌─────────────────────┐
│  FastAPI Python     │  AI Engineer
│  Inference API      │  TensorFlow · Scikit-learn
└──────────┬──────────┘
           │ loads
           ▼
┌─────────────────────┐
│  Trained Model      │  pulsevera_dl_model.keras
│  + Preprocessing    │  scaler.pkl · threshold.pkl
└─────────────────────┘
```

---

## Handoff Antar Role

### DS → AI Engineer ✅
| Deliverable | Lokasi | Status |
|---|---|---|
| X_train, X_test, y_train, y_test | `data/final/` | ✅ Siap |
| dataset_final.csv (+ 6 fitur baru) | `data/final/` | ✅ Siap |
| data_dictionary.md (47 kolom) | root | ✅ Siap |

### AI Engineer → Full-Stack ✅
| Deliverable | Lokasi | Status |
|---|---|---|
| FastAPI endpoint `/api/v1/predict` | `ml-api/main.py` | ✅ Siap |
| Format request & response JSON | README ini | ✅ Terdokumentasi |
| Model tersimpan format produksi | `ml-api/models/` | ✅ Siap |

---

*Dataset: CDC BRFSS 2022 | 445.132 responden | Coding Camp 2026 powered by DBS Foundation*
