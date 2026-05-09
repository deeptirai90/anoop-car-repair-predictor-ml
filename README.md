# 🚗 Car Repair Cost Predictor — Automotive ML

AI-powered repair cost prediction based on incident speed and parameters.
Includes 10 synthetic car incident images across severity levels.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_dataset.py

# 3. Generate 10 incident images
python generate_car_images.py

# 4. Train ML models
python train_model.py --compare

# 5. Launch Streamlit app
streamlit run app.py
```

Open → **http://localhost:8501**

---

## 📦 Project Structure

```
car_repair_predictor/
├── app.py                        ← Streamlit app (5 tabs)
├── train_model.py                ← ML training pipeline
├── generate_dataset.py           ← Synthetic dataset generator
├── generate_car_images.py        ← 10 incident image generator
├── requirements.txt
├── README.md
├── data/
│   ├── car_incidents_train.csv   ← 800 training records
│   ├── car_incidents_test.csv    ← 200 test records
│   └── images/
│       ├── incident_01_speed_15kmh.png
│       ├── incident_02_speed_25kmh.png
│       ├── incident_03_speed_35kmh.png
│       ├── incident_04_speed_45kmh.png
│       ├── incident_05_speed_55kmh.png
│       ├── incident_06_speed_65kmh.png
│       ├── incident_07_speed_75kmh.png
│       ├── incident_08_speed_85kmh.png
│       ├── incident_09_speed_100kmh.png
│       └── incident_10_speed_120kmh.png
└── models/
    ├── best_model.pkl            ← Trained Gradient Boosting model
    ├── scaler.pkl                ← StandardScaler
    ├── features.pkl              ← Feature list
    └── report.json               ← Training metrics
```

---

## 🧠 ML Models

| Model | R² | RMSE |
|---|---|---|
| **Gradient Boosting** ✅ | **0.9275** | ₹1,16,098 |
| Linear Regression | 0.8888 | ₹1,43,791 |
| Ridge Regression | 0.8883 | ₹1,44,142 |
| Random Forest | 0.8146 | ₹1,85,711 |
| Decision Tree | 0.5898 | ₹2,76,194 |

---

## 📊 Features Used (16 total)

| Feature | Type | Description |
|---|---|---|
| speed_kmh | Numeric | Impact speed in km/h |
| vehicle_age_years | Numeric | Age of vehicle |
| vehicle_value_inr | Numeric | Market value in ₹ |
| occupants | Numeric | Number of occupants |
| airbag_deployed | Binary | Airbag triggered |
| seatbelt_worn | Binary | Seatbelt used |
| previous_accidents | Numeric | Prior accident count |
| driver_age | Numeric | Driver age |
| collision_enc | Encoded | Collision type severity |
| road_enc | Encoded | Road condition factor |
| weather_enc | Encoded | Weather risk factor |
| vehicle_type_enc | Encoded | Vehicle class factor |
| tyre_enc | Encoded | Tyre condition factor |
| speed_squared | Engineered | speed² |
| speed_x_collision | Engineered | speed × collision_enc |
| age_x_speed | Engineered | vehicle_age × speed |

---

## 🖼️ Incident Images

10 synthetic images generated using Python PIL showing:
- Top-down car silhouette
- Damage cracks radiating from impact point
- Speed gauge bar
- Severity badge (Minor → Total Loss)
- Smoke & fire effects for high-speed incidents
- Debris scatter particles
- Cost & parts info overlay

---

## train_model.py Commands

```bash
python train_model.py                  # train all 5 models
python train_model.py --model gb       # train Gradient Boosting only
python train_model.py --compare        # full comparison table
python train_model.py --predict        # sample prediction after training
python train_model.py --model rf --predict  # train RF + predict
python train_model.py --no-save        # train without saving files
```

---

## 💰 Cost Formula (synthetic data)

```
repair_cost = (
    speed² × 18
  + vehicle_value × 0.08
  + vehicle_age × 2500
  + airbag_deployed × 45000
  + prev_accidents × 5000
  - seatbelt_worn × 8000
  + (70 - driver_age) × 400
  + noise
) × collision_factor × road_factor × weather_factor × vehicle_factor × tyre_factor
```

---

## 📱 Streamlit App Tabs

1. **Dashboard** — Metrics, cost distribution, scatter, collision chart
2. **Predict Cost** — Interactive form → instant cost estimate + parts damaged
3. **Incident Gallery** — All 10 incident images with metadata
4. **Model Analysis** — Actual vs predicted, residuals, feature importance
5. **Dataset** — View + download train/test CSV

---

## ☁️ Deploy to Streamlit Cloud (Free)

```bash
git init && git add . && git commit -m "Car repair ML"
git remote add origin https://github.com/YOUR_USER/car-repair-ml.git
git push -u origin main
# Go to share.streamlit.io → New App → Deploy
```
