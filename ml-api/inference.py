"""
Pulsevera simple inference module.

This file is intentionally independent from FastAPI so the saved DL model can
be tested from a Python script before it is wired into `main.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

DL_MODEL_PATH = MODELS_DIR / "pulsevera_dl_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"


DEFAULT_VALUES: dict[str, float] = {
    "PhysicalHealthDays": 0.0,
    "MentalHealthDays": 0.0,
    "LastCheckupTime": 3.0,
    "RemovedTeeth": 3.0,
    "HadAngina": 0.0,
    "HadStroke": 0.0,
    "HadAsthma": 0.0,
    "HadSkinCancer": 0.0,
    "HadCOPD": 0.0,
    "HadDepressiveDisorder": 0.0,
    "HadKidneyDisease": 0.0,
    "HadArthritis": 0.0,
    "DeafOrHardOfHearing": 0.0,
    "BlindOrVisionDifficulty": 0.0,
    "DifficultyConcentrating": 0.0,
    "DifficultyWalking": 0.0,
    "DifficultyDressingBathing": 0.0,
    "DifficultyErrands": 0.0,
    "ECigaretteUsage": 0.0,
    "ChestScan": 0.0,
    "HIVTesting": 0.0,
    "FluVaxLast12": 1.0,
    "HighRiskLastYear": 0.0,
    "CovidPos": 0.0,
    "Race_Black only, Non-Hispanic": 0.0,
    "Race_Hispanic": 0.0,
    "Race_Multiracial, Non-Hispanic": 0.0,
    "Race_Other race only, Non-Hispanic": 0.0,
    "Race_White only, Non-Hispanic": 1.0,
}


SEX_MAPPING = {
    "Female": 0,
    "Perempuan": 0,
    "Male": 1,
    "Laki-laki": 1,
}

YES_NO_MAPPING = {
    "No": 0,
    "Tidak": 0,
    "Yes": 1,
    "Ya": 1,
}

SMOKER_MAPPING = {
    "Never": 0,
    "Tidak pernah": 0,
    "Former": 1,
    "Mantan": 1,
    "Current-some": 2,
    "Perokok (kadang)": 2,
    "Current-every": 3,
    "Perokok (setiap hari)": 3,
}

GENERAL_HEALTH_MAPPING = {
    "Poor": 1,
    "Buruk": 1,
    "Fair": 2,
    "Cukup": 2,
    "Good": 3,
    "Baik": 3,
    "Very good": 4,
    "Sangat Baik": 4,
    "Excellent": 5,
    "Sangat Baik Sekali": 5,
}

DIABETES_MAPPING = {
    "No": 0,
    "Tidak": 0,
    "Pre-diabetes": 1,
    "Yes": 3,
    "Ya": 3,
}


class PulseveraInference:
    """Load Pulsevera artifacts and run one-row DL inference."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.model = tf.keras.models.load_model(
            self.models_dir / DL_MODEL_PATH.name,
            compile=False,
        )
        self.scaler = joblib.load(self.models_dir / SCALER_PATH.name)
        self.feature_order: list[str] = joblib.load(self.models_dir / FEATURE_ORDER_PATH.name)
        self.threshold = float(joblib.load(self.models_dir / THRESHOLD_PATH.name))

    def preprocess(self, user_input: dict[str, Any]) -> pd.DataFrame:
        """Convert 10 web-form fields into the 46 model features."""
        data = DEFAULT_VALUES.copy()

        height = float(user_input["height_meters"])
        weight = float(user_input["weight_kg"])
        bmi = weight / (height**2)

        data["Sex"] = self._map_value(user_input["sex"], SEX_MAPPING, "sex")
        data["AgeCategory"] = float(user_input["age_category"])
        data["HeightInMeters"] = height
        data["WeightInKilograms"] = weight
        data["BMI"] = bmi
        data["SleepHours"] = float(user_input["sleep_hours"])
        data["PhysicalActivities"] = self._map_value(
            user_input["physical_activities"],
            YES_NO_MAPPING,
            "physical_activities",
        )
        data["AlcoholDrinkers"] = self._map_value(user_input["alcohol"], YES_NO_MAPPING, "alcohol")
        data["SmokerStatus"] = self._map_value(
            user_input["smoker_status"],
            SMOKER_MAPPING,
            "smoker_status",
        )
        data["GeneralHealth"] = self._map_value(
            user_input["general_health"],
            GENERAL_HEALTH_MAPPING,
            "general_health",
        )
        data["HadDiabetes"] = self._map_value(
            user_input.get("diabetes", "No"),
            DIABETES_MAPPING,
            "diabetes",
        )

        data["IsActiveSmoker"] = float(data["SmokerStatus"] >= 2)
        data["IsObese"] = float(data["BMI"] >= 30)
        data["IsSleepDeprived"] = float(data["SleepHours"] < 6)
        data["LifestyleRiskScore"] = (
            data["IsActiveSmoker"]
            + (1.0 - data["PhysicalActivities"])
            + data["AlcoholDrinkers"]
            + data["IsSleepDeprived"]
            + data["IsObese"]
        )
        data["HasChronicCondition"] = float(
            any(
                data[column] > 0
                for column in [
                    "HadDiabetes",
                    "HadStroke",
                    "HadAsthma",
                    "HadCOPD",
                    "HadKidneyDisease",
                ]
            )
        )
        data["PoorHealthDays_Total"] = data["PhysicalHealthDays"] + data["MentalHealthDays"]

        missing = [feature for feature in self.feature_order if feature not in data]
        if missing:
            raise ValueError(f"Missing features after preprocessing: {missing}")

        return pd.DataFrame([data], columns=self.feature_order).astype("float32")

    def predict(self, user_input: dict[str, Any]) -> dict[str, Any]:
        """Return risk probability and label for one user input."""
        input_df = self.preprocess(user_input)
        scaled_input = self.scaler.transform(input_df).astype("float32")
        probability = float(np.ravel(self.model.predict(scaled_input, verbose=0))[0])

        return {
            "risk_score": round(probability, 4),
            "risk_label": self._risk_label(probability),
            "threshold": self.threshold,
            "model": "DL",
        }

    @staticmethod
    def _map_value(value: Any, mapping: dict[str, int], field_name: str) -> float:
        if value not in mapping:
            allowed = ", ".join(sorted(mapping))
            raise ValueError(f"Invalid value for {field_name}: {value!r}. Allowed: {allowed}")
        return float(mapping[value])

    def _risk_label(self, probability: float) -> str:
        if probability < self.threshold:
            return "Rendah"
        if probability < 0.6:
            return "Sedang"
        return "Tinggi"


def predict_risk(user_input: dict[str, Any]) -> dict[str, Any]:
    """Convenience function for quick one-off inference."""
    return PulseveraInference().predict(user_input)


if __name__ == "__main__":
    print("=== Pulsevera — Cek Risiko Penyakit Jantung ===\n")

    # Jenis kelamin
    print("Jenis kelamin:")
    print("  1. Laki-laki")
    print("  2. Perempuan")
    sex = "Male" if input("> ").strip() == "1" else "Female"

    # Kategori usia
    print("\nKategori usia:")
    age_labels = [
        "1. 18-24", "2. 25-29", "3. 30-34", "4. 35-39", "5. 40-44",
        "6. 45-49", "7. 50-54", "8. 55-59", "9. 60-64", "10. 65-69",
        "11. 70-74", "12. 75-79", "13. 80+"
    ]
    for label in age_labels:
        print(f"  {label}")
    age_category = int(input("> ").strip())

    # Tinggi dan berat
    print("\nTinggi badan (meter, contoh: 1.70):")
    height_meters = float(input("> ").strip())

    print("Berat badan (kg, contoh: 70):")
    weight_kg = float(input("> ").strip())

    # Jam tidur
    print("\nRata-rata jam tidur per malam (contoh: 7):")
    sleep_hours = float(input("> ").strip())

    # Aktivitas fisik
    print("\nAktif berolahraga?")
    print("  1. Ya")
    print("  2. Tidak")
    physical_activities = "Yes" if input("> ").strip() == "1" else "No"

    # Status merokok
    print("\nStatus merokok:")
    print("  1. Tidak pernah")
    print("  2. Mantan perokok")
    print("  3. Perokok (kadang-kadang)")
    print("  4. Perokok (setiap hari)")
    smoker_map = {"1": "Never", "2": "Former", "3": "Current-some", "4": "Current-every"}
    smoker_status = smoker_map.get(input("> ").strip(), "Never")

    # Alkohol
    print("\nMengonsumsi alkohol?")
    print("  1. Ya")
    print("  2. Tidak")
    alcohol = "Yes" if input("> ").strip() == "1" else "No"

    # Kesehatan umum
    print("\nKondisi kesehatan umum:")
    print("  1. Buruk")
    print("  2. Cukup")
    print("  3. Baik")
    print("  4. Sangat Baik")
    print("  5. Sangat Baik Sekali")
    health_map = {"1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent"}
    general_health = health_map.get(input("> ").strip(), "Good")

    # Diabetes
    print("\nRiwayat diabetes:")
    print("  1. Tidak")
    print("  2. Pre-diabetes")
    print("  3. Ya")
    diabetes_map = {"1": "No", "2": "Pre-diabetes", "3": "Yes"}
    diabetes = diabetes_map.get(input("> ").strip(), "No")

    user_input = {
        "sex"                 : sex,
        "age_category"        : age_category,
        "height_meters"       : height_meters,
        "weight_kg"           : weight_kg,
        "sleep_hours"         : sleep_hours,
        "physical_activities" : physical_activities,
        "smoker_status"       : smoker_status,
        "alcohol"             : alcohol,
        "general_health"      : general_health,
        "diabetes"            : diabetes,
    }

    result = predict_risk(user_input)

    print("\n=== HASIL PREDIKSI ===")
    print(f"Skor Risiko  : {result['risk_score'] * 100:.1f}%")
    print(f"Label Risiko : {result['risk_label']}")
