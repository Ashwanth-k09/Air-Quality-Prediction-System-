"""Command-line air quality diagnosis using the trained model."""
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_air_quality.pkl"

# (column, prompt, safe threshold, moderate threshold, health note)
INPUTS = [
    ("PT08.S1(CO)", "CO sensor (PT08.S1) [uV]", 1500, 1300,
     "High CO can reduce oxygen in the blood and stress the heart."),
    ("C6H6(GT)", "Benzene (C6H6) [ug/m3]", 5, 3,
     "Benzene is a carcinogen; long exposure affects bone marrow."),
    ("PT08.S2(NMHC)", "NMHC sensor (PT08.S2) [uV]", 1200, 1000,
     "High VOC levels can signal toxic conditions and affect breathing."),
    ("NOx(GT)", "Nitric oxides (NOx) [ppb]", 100, 80,
     "NOx can aggravate asthma and reduce lung function."),
    ("PT08.S3(NOx)", "NOx sensor (PT08.S3) [uV]", 1500, 1300,
     "High NOx sensor values reflect combustion pollution."),
    ("NO2(GT)", "Nitrogen dioxide (NO2) [ug/m3]", 100, 75,
     "Elevated NO2 causes respiratory irritation."),
    ("PT08.S4(NO2)", "NO2 sensor (PT08.S4) [uV]", 1500, 1300,
     "High NO2 sensor values signal air toxicity."),
    ("PT08.S5(O3)", "Ozone sensor (PT08.S5) [uV]", 1200, 1000,
     "Ground-level ozone can harm lungs and trigger asthma."),
    ("T", "Temperature [C]", 35, 30,
     "High temperature intensifies the effect of pollutants."),
    ("RH", "Relative humidity [%]", 80, 60,
     "High humidity helps allergens and particles stay suspended."),
    ("AH", "Absolute humidity [g/m3]", 1.2, 0.9,
     "High absolute humidity increases pollutant absorption."),
]


def ask(prompt: str) -> float:
    while True:
        try:
            return float(input(f"{prompt}: "))
        except ValueError:
            print("Invalid entry. Please enter a number.")


def main() -> None:
    model = joblib.load(MODEL_PATH)
    print("\nAir Quality Diagnosis System")
    print("Enter the sensor readings one by one:\n")

    values, above_safe, above_moderate = {}, [], []
    for col, prompt, safe, moderate, note in INPUTS:
        val = ask(prompt)
        values[col] = val
        if val >= safe:
            above_safe.append((prompt, val, safe, note))
        elif val >= moderate:
            above_moderate.append((prompt, val, moderate, note))

    now = datetime.now()
    values["Hour"], values["Month"] = now.hour, now.month

    row = pd.DataFrame([values])[list(model.feature_names_in_)]
    prediction = model.predict(row)[0]

    if above_safe or prediction == 1:
        status = "BAD"
    elif above_moderate:
        status = "MODERATE"
    else:
        status = "GOOD"

    print(f"\nModel prediction: {'bad air' if prediction == 1 else 'not bad'}")
    print(f"Air quality classification: {status}\n")

    if status == "BAD":
        for i, (label, val, limit, note) in enumerate(above_safe, 1):
            print(f"{i}. {label}: {val} >= {limit}\n   -> {note}")
        print("\nHealth warning: limit outdoor activity; wear an N95 mask if possible.")
    elif status == "MODERATE":
        for i, (label, val, limit, note) in enumerate(above_moderate, 1):
            print(f"{i}. {label}: {val} >= {limit}\n   -> {note}")
        print("\nAdvisory: sensitive groups may feel symptoms; limit outdoor exercise.")
    else:
        print("All indicators are within acceptable ranges.")


if __name__ == "__main__":
    main()