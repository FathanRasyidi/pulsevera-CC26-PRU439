"""
Pulsevera — ML Inference API
FastAPI endpoint untuk prediksi risiko penyakit jantung.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from inference import PulseveraInference

app = FastAPI(
    title="Pulsevera ML API",
    description="API prediksi risiko penyakit jantung — Pulsevera CC26-PRU439",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model sekali saat server start — tidak perlu load ulang setiap request
inferencer = None

@app.on_event("startup")
def load_model():
    global inferencer
    inferencer = PulseveraInference()
    print("Model berhasil dimuat.")


# ── Schema ────────────────────────────────────────────────────────
class UserInput(BaseModel):
    sex: str                 = Field(..., example="Male")
    age_category: int        = Field(..., ge=1, le=13, example=7)
    height_meters: float     = Field(..., ge=1.0, le=2.5, example=1.70)
    weight_kg: float         = Field(..., ge=30, le=200, example=70.0)
    sleep_hours: float       = Field(..., ge=1, le=14, example=7.0)
    physical_activities: str = Field(..., example="Yes")
    smoker_status: str       = Field(..., example="Never")
    alcohol: str             = Field(..., example="No")
    general_health: str      = Field(..., example="Good")
    diabetes: Optional[str]  = Field(default="No", example="No")


class PredictionResult(BaseModel):
    risk_score: float    # probabilitas 0.0–1.0
    risk_label: str      # "Rendah" / "Sedang" / "Tinggi"
    top_risk_factors: list
    recommendations: list


# ── Rekomendasi ───────────────────────────────────────────────────
RECOMMENDATIONS = {
    "IsActiveSmoker"    : "Berhenti merokok — konsultasikan program berhenti merokok ke dokter.",
    "IsObese"           : "Turunkan berat badan dengan diet seimbang dan olahraga rutin.",
    "IsSleepDeprived"   : "Tingkatkan jam tidur menjadi 7–8 jam per malam.",
    "PhysicalActivities": "Olahraga minimal 30 menit per hari, 5 hari seminggu.",
    "AlcoholDrinkers"   : "Kurangi atau hentikan konsumsi alkohol.",
    "HadDiabetes"       : "Kontrol gula darah secara rutin dan konsultasi ke dokter.",
    "AgeCategory"       : "Lakukan pemeriksaan jantung rutin setiap tahun.",
    "BMI"               : "Jaga berat badan ideal dengan pola makan sehat.",
    "GeneralHealth"     : "Lakukan medical check-up menyeluruh secara berkala.",
    "SmokerStatus"      : "Berhenti merokok — risiko jantung turun signifikan setelah berhenti.",
}

def get_recommendations(user_input: dict, top_factors: list) -> list:
    result = []
    for factor in top_factors:
        if factor in RECOMMENDATIONS:
            result.append(RECOMMENDATIONS[factor])

    # Tambahkan rekomendasi spesifik berdasarkan input user
    if user_input.get("physical_activities") in ("No", "Tidak") and "Olahraga" not in str(result):
        result.append("Olahraga minimal 30 menit per hari, 5 hari seminggu.")
    if user_input.get("sleep_hours", 8) < 6 and "tidur" not in str(result):
        result.append("Tingkatkan jam tidur menjadi 7–8 jam per malam.")

    return result[:3] if result else ["Jaga pola hidup sehat dan lakukan pemeriksaan rutin."]


def get_top_risk_factors(user_input: dict) -> list:
    # Tentukan faktor risiko berdasarkan input user secara langsung
    # Lebih sederhana dan transparan dibanding SHAP untuk API ini
    factors = []

    bmi = float(user_input["weight_kg"]) / (float(user_input["height_meters"]) ** 2)
    smoker = user_input.get("smoker_status", "Never")
    sleep  = float(user_input.get("sleep_hours", 8))
    active = user_input.get("physical_activities", "Yes")
    health = user_input.get("general_health", "Good")
    age    = int(user_input.get("age_category", 1))

    if smoker in ("Current-some", "Current-every"):
        factors.append("IsActiveSmoker")
    if bmi >= 30:
        factors.append("IsObese")
    if age >= 9:  # 60 tahun ke atas
        factors.append("AgeCategory")
    if sleep < 6:
        factors.append("IsSleepDeprived")
    if active in ("No", "Tidak"):
        factors.append("PhysicalActivities")
    if health in ("Poor", "Fair"):
        factors.append("GeneralHealth")
    if user_input.get("diabetes", "No") not in ("No", "Tidak"):
        factors.append("HadDiabetes")
    if user_input.get("alcohol", "No") in ("Yes", "Ya"):
        factors.append("AlcoholDrinkers")

    return factors[:3] if factors else ["GeneralHealth"]


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": inferencer is not None}


@app.post("/api/v1/predict", response_model=PredictionResult)
async def predict(user_input: UserInput):
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    try:
        input_dict = user_input.model_dump()
        result     = inferencer.predict(input_dict)
        factors    = get_top_risk_factors(input_dict)
        recs       = get_recommendations(input_dict, factors)

        return PredictionResult(
            risk_score       = result["risk_score"],
            risk_label       = result["risk_label"],
            top_risk_factors = factors,
            recommendations  = recs,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
