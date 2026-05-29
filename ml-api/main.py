"""
Pulsevera — ML Inference API
FastAPI endpoint untuk prediksi risiko penyakit jantung.
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from inference import PulseveraInference

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

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
gemini_client = None

GEMINI_API_KEY_PLACEHOLDER = "PASTE_YOUR_GEMINI_API_KEY_HERE"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY_PLACEHOLDER)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.0-flash")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() == "true"


def get_gemini_inactive_reason() -> Optional[str]:
    if not GEMINI_ENABLED:
        return "GEMINI_ENABLED=false"
    if genai is None or types is None:
        return "package google-genai belum terpasang di environment Python ini"

    api_key = (GEMINI_API_KEY or "").strip()
    if not api_key or api_key == GEMINI_API_KEY_PLACEHOLDER:
        return "GEMINI_API_KEY belum diisi atau masih placeholder"

    return None

@app.on_event("startup")
def load_model():
    global inferencer, gemini_client
    inferencer = PulseveraInference()
    print("Model berhasil dimuat.")

    gemini_client = create_gemini_client()
    if gemini_client is None:
        reason = get_gemini_inactive_reason() or "client Gemini tidak tersedia"
        print(f"Gemini belum aktif ({reason}). Rekomendasi memakai fallback rule-based.")
    else:
        print(f"Gemini aktif untuk rekomendasi. Model: {GEMINI_MODEL}")


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
    recommendation_source: str  # "gemini" / "rule_based"


# ── Rekomendasi ───────────────────────────────────────────────────
class GeminiRecommendationResponse(BaseModel):
    recommendations: list[str]


AGE_CATEGORY_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

RISK_FACTOR_LABELS = {
    "IsActiveSmoker": "masih merokok aktif",
    "IsObese": "BMI masuk kategori obesitas",
    "IsSleepDeprived": "durasi tidur kurang dari 6 jam",
    "PhysicalActivities": "belum rutin beraktivitas fisik",
    "AlcoholDrinkers": "mengonsumsi alkohol",
    "HadDiabetes": "memiliki riwayat diabetes atau pra-diabetes",
    "AgeCategory": "usia lebih tinggi",
    "BMI": "berat badan perlu dikelola",
    "GeneralHealth": "kesehatan umum kurang optimal",
    "SmokerStatus": "status merokok meningkatkan risiko",
}

GEMINI_RECOMMENDATION_PROMPT = """
Anda adalah asisten rekomendasi gaya hidup untuk Pulsevera, aplikasi edukasi
risiko penyakit jantung. Buat rekomendasi yang personal berdasarkan profil
pengguna dan faktor risiko prioritas yang diberikan.

Aturan wajib:
- Jawab dalam Bahasa Indonesia.
- Output hanya JSON object valid dengan key "recommendations".
- Isi "recommendations" harus berupa tepat 3 string.
- Setiap rekomendasi maksimal 22 kata, konkret, aman, dan dapat dilakukan.
- Jangan menyebut skor, probabilitas, model, machine learning, atau deep learning.
- Jangan mendiagnosis penyakit dan jangan memberi dosis obat atau suplemen.
- Jika ada faktor diabetes, perokok aktif, obesitas, usia 60+, atau kesehatan umum buruk, anjurkan konsultasi tenaga kesehatan.
- Hindari kalimat menakut-nakuti; gunakan nada profesional dan suportif.
"""

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

def create_gemini_client():
    if get_gemini_inactive_reason() is not None:
        return None

    api_key = (GEMINI_API_KEY or "").strip()
    return genai.Client(api_key=api_key)


def get_gemini_model_candidates() -> list[str]:
    model_names = [GEMINI_MODEL]
    model_names.extend(
        model.strip()
        for model in GEMINI_FALLBACK_MODELS.split(",")
        if model.strip()
    )

    deduped_models = []
    for model in model_names:
        if model not in deduped_models:
            deduped_models.append(model)

    return deduped_models


def build_recommendation_prompt(user_input: dict[str, Any], top_factors: list[str]) -> str:
    bmi = float(user_input["weight_kg"]) / (float(user_input["height_meters"]) ** 2)
    age_category = int(user_input.get("age_category", 1))

    profile = {
        "jenis_kelamin": user_input.get("sex"),
        "rentang_usia": AGE_CATEGORY_LABELS.get(age_category, f"Kategori {age_category}"),
        "bmi": round(bmi, 1),
        "jam_tidur": user_input.get("sleep_hours"),
        "aktivitas_fisik": user_input.get("physical_activities"),
        "status_merokok": user_input.get("smoker_status"),
        "alkohol": user_input.get("alcohol"),
        "kesehatan_umum": user_input.get("general_health"),
        "diabetes": user_input.get("diabetes", "No"),
    }
    readable_factors = [RISK_FACTOR_LABELS.get(factor, factor) for factor in top_factors]

    # Rekomendasi memakai profil input dan faktor risiko rule-based, bukan skor prediksi DL.
    return (
        "Buat 3 rekomendasi gaya hidup untuk profil berikut.\n"
        f"Profil pengguna:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n"
        f"Faktor risiko prioritas:\n{json.dumps(readable_factors, ensure_ascii=False)}\n"
        "Fokus pada perubahan perilaku yang paling relevan untuk faktor risiko tersebut."
    )


def build_gemini_generation_config(model_name: str):
    config = {
        "system_instruction": GEMINI_RECOMMENDATION_PROMPT,
        "temperature": 0.4,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
        "response_schema": GeminiRecommendationResponse,
    }

    # Gemini 2.5 dapat memakai token untuk thinking; matikan agar respons JSON pendek tidak terpotong.
    if "2.5" in model_name:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    return types.GenerateContentConfig(**config)


def normalize_recommendations(recommendations: Any) -> list[str]:
    if not isinstance(recommendations, list):
        return []

    clean_recommendations = []
    for item in recommendations:
        if isinstance(item, str) and item.strip():
            clean_recommendations.append(item.strip())

    return clean_recommendations[:3]


def parse_gemini_recommendations(response: Any) -> list[str]:
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, GeminiRecommendationResponse):
        return normalize_recommendations(parsed.recommendations)

    if isinstance(parsed, dict):
        return normalize_recommendations(parsed.get("recommendations"))

    raw_text = (getattr(response, "text", None) or "").strip()
    if not raw_text:
        return []

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    recommendations = payload.get("recommendations") if isinstance(payload, dict) else payload
    return normalize_recommendations(recommendations)


def get_gemini_recommendations(user_input: dict[str, Any], top_factors: list[str]) -> list[str]:
    if gemini_client is None or types is None:
        return []

    for model_name in get_gemini_model_candidates():
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=build_recommendation_prompt(user_input, top_factors),
                config=build_gemini_generation_config(model_name),
            )
            recommendations = parse_gemini_recommendations(response)
            if recommendations:
                if model_name != GEMINI_MODEL:
                    print(f"Gemini rekomendasi memakai fallback model: {model_name}")
                return recommendations

            print(f"Gemini model {model_name} mengembalikan rekomendasi kosong.")
        except Exception as exc:
            print(f"Gemini model {model_name} gagal: {exc}")

    print("Semua model Gemini gagal, fallback rule-based dipakai.")
    return []


def get_rule_based_recommendations(user_input: dict, top_factors: list) -> list:
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


def get_recommendations(user_input: dict[str, Any], top_factors: list[str]) -> tuple[list[str], str]:
    ai_recommendations = get_gemini_recommendations(user_input, top_factors)
    if ai_recommendations:
        return ai_recommendations, "gemini"

    return get_rule_based_recommendations(user_input, top_factors), "rule_based"


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
    gemini_active = gemini_client is not None
    return {
        "status": "ok",
        "model_loaded": inferencer is not None,
        "gemini_recommendations": gemini_active,
        "gemini_model": GEMINI_MODEL,
        "gemini_model_candidates": get_gemini_model_candidates(),
        "gemini_inactive_reason": None if gemini_active else get_gemini_inactive_reason(),
    }


@app.post("/api/v1/predict", response_model=PredictionResult)
async def predict(user_input: UserInput):
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    try:
        input_dict = user_input.model_dump()
        result     = inferencer.predict(input_dict)
        factors    = get_top_risk_factors(input_dict)
        recs, rec_source = get_recommendations(input_dict, factors)

        return PredictionResult(
            risk_score       = result["risk_score"],
            risk_label       = result["risk_label"],
            top_risk_factors = factors,
            recommendations  = recs,
            recommendation_source = rec_source,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
