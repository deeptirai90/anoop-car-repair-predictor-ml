"""
generate_dataset.py
Creates data/car_incidents_train.csv and data/car_incidents_test.csv
with 1000 synthetic car accident records.
"""

import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

def generate(n=1000, seed=42):
    np.random.seed(seed)

    # Core features
    speed              = np.random.randint(5, 141, n)
    vehicle_age        = np.random.randint(0, 21, n)
    vehicle_value      = np.random.randint(300000, 3000001, n)
    occupants          = np.random.randint(1, 6, n)
    airbag_deployed    = (speed > 40).astype(int) + np.random.binomial(1, 0.1, n)
    airbag_deployed    = np.clip(airbag_deployed, 0, 1)
    collision_type     = np.random.choice(["Head-on","Rear-end","Side","Rollover","Pole"], n,
                                           p=[0.25, 0.30, 0.25, 0.10, 0.10])
    road_condition     = np.random.choice(["Dry","Wet","Icy","Gravel"], n, p=[0.50,0.30,0.10,0.10])
    weather            = np.random.choice(["Clear","Rain","Fog","Snow"], n, p=[0.55,0.25,0.12,0.08])
    vehicle_type       = np.random.choice(["Sedan","SUV","Hatchback","Truck","Two-wheeler"], n,
                                           p=[0.35,0.25,0.20,0.10,0.10])
    insurance_type     = np.random.choice(["Comprehensive","Third-party","None"], n, p=[0.55,0.35,0.10])
    seatbelt_worn      = np.random.binomial(1, 0.75, n)
    previous_accidents = np.random.randint(0, 5, n)
    driver_age         = np.random.randint(18, 71, n)
    tyre_condition     = np.random.choice(["Good","Worn","Bald"], n, p=[0.60,0.28,0.12])

    # Encodings
    coll_enc  = {"Head-on":1.8,"Rear-end":1.0,"Side":1.3,"Rollover":2.2,"Pole":1.5}
    road_enc  = {"Dry":1.0,"Wet":1.3,"Icy":1.8,"Gravel":1.2}
    weath_enc = {"Clear":1.0,"Rain":1.25,"Fog":1.15,"Snow":1.4}
    vtype_enc = {"Sedan":1.0,"SUV":1.3,"Hatchback":0.85,"Truck":1.5,"Two-wheeler":0.6}
    tyre_enc  = {"Good":1.0,"Worn":1.15,"Bald":1.35}

    ce = np.array([coll_enc[c] for c in collision_type])
    re = np.array([road_enc[r] for r in road_condition])
    we = np.array([weath_enc[w] for w in weather])
    ve = np.array([vtype_enc[v] for v in vehicle_type])
    te = np.array([tyre_enc[t] for t in tyre_condition])

    noise = np.random.normal(0, 8000, n)

    # Repair cost formula
    repair_cost = (
        speed**2 * 18
        + vehicle_value * 0.08
        + vehicle_age   * 2500
        + airbag_deployed * 45000
        + previous_accidents * 5000
        - seatbelt_worn * 8000
        + (70 - driver_age.clip(18,70)) * 400
        + noise
    ) * ce * re * we * ve * te

    repair_cost = np.clip(repair_cost, 5000, None).round(-2)

    # Damage severity label
    def severity(s):
        if s < 30:  return "Minor"
        if s < 60:  return "Moderate"
        if s < 90:  return "Severe"
        if s < 110: return "Critical"
        return "Total Loss"

    damage_severity = [severity(s) for s in speed]

    # Parts damaged
    def parts(s, ct):
        p = []
        if s > 10:  p.append("Bumper")
        if s > 25:  p.append("Hood" if ct in ["Head-on","Pole"] else "Door Panel")
        if s > 40:  p.append("Headlights/Taillights")
        if s > 55:  p.append("Radiator")
        if s > 70:  p.append("Airbags"); p.append("Frame")
        if s > 90:  p.append("Engine")
        if s > 110: p.append("Total Loss")
        return ", ".join(p) if p else "Minor scratches"

    parts_damaged = [parts(s, c) for s, c in zip(speed, collision_type)]

    df = pd.DataFrame({
        "incident_id":       range(1, n+1),
        "speed_kmh":         speed,
        "vehicle_age_years": vehicle_age,
        "vehicle_value_inr": vehicle_value,
        "occupants":         occupants,
        "airbag_deployed":   airbag_deployed,
        "collision_type":    collision_type,
        "road_condition":    road_condition,
        "weather":           weather,
        "vehicle_type":      vehicle_type,
        "insurance_type":    insurance_type,
        "seatbelt_worn":     seatbelt_worn,
        "previous_accidents":previous_accidents,
        "driver_age":        driver_age,
        "tyre_condition":    tyre_condition,
        "collision_enc":     ce.round(2),
        "road_enc":          re.round(2),
        "weather_enc":       we.round(2),
        "vehicle_type_enc":  ve.round(2),
        "tyre_enc":          te.round(2),
        "damage_severity":   damage_severity,
        "parts_damaged":     parts_damaged,
        "repair_cost_inr":   repair_cost.astype(int),
    })

    return df

if __name__ == "__main__":
    df    = generate(1000)
    train = df.sample(frac=0.8, random_state=42)
    test  = df.drop(train.index)
    train.to_csv("data/car_incidents_train.csv", index=False)
    test.to_csv("data/car_incidents_test.csv",   index=False)
    print(f"✅  Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"\nRepair cost stats:")
    print(df["repair_cost_inr"].describe().apply(lambda x: f"₹{x:,.0f}"))
    print(f"\nSeverity distribution:")
    print(df["damage_severity"].value_counts())
