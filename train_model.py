"""
train_model.py
==============
ML pipeline to predict car repair cost from incident features.

Usage
-----
    python train_model.py               # train all models
    python train_model.py --model rf    # train one model
    python train_model.py --compare     # full comparison
    python train_model.py --predict     # interactive prediction
"""

import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

# ANSI colours
G="\033[92m"; Y="\033[93m"; C="\033[96m"; R="\033[91m"; B="\033[1m"; X="\033[0m"

def banner(t, c=C): print(f"\n{c}{B}{'═'*64}\n  {t}\n{'═'*64}{X}")
def sec(t):         print(f"\n{Y}{B}── {t} {'─'*(58-len(t))}{X}")

TRAIN_CSV   = "data/car_incidents_train.csv"
TEST_CSV    = "data/car_incidents_test.csv"
MODEL_DIR   = "models"

FEATURE_COLS = [
    "speed_kmh","vehicle_age_years","vehicle_value_inr","occupants",
    "airbag_deployed","seatbelt_worn","previous_accidents","driver_age",
    "collision_enc","road_enc","weather_enc","vehicle_type_enc","tyre_enc",
    # engineered
    "speed_squared","speed_x_collision","age_x_speed",
]

MODEL_REGISTRY = {
    "lr":   ("Linear Regression",  LinearRegression()),
    "ridge":("Ridge Regression",   Ridge(alpha=10)),
    "dt":   ("Decision Tree",      DecisionTreeRegressor(max_depth=8, random_state=42)),
    "rf":   ("Random Forest",      RandomForestRegressor(n_estimators=200, max_depth=14,
                                       min_samples_leaf=2, random_state=42, n_jobs=-1)),
    "gb":   ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                       learning_rate=0.06, subsample=0.8, random_state=42)),
}
LINEAR = {"lr","ridge"}

def load():
    sec("Loading data")
    for p in [TRAIN_CSV, TEST_CSV]:
        if not os.path.exists(p):
            print(f"{R}✗ {p} not found. Run: python generate_dataset.py{X}"); sys.exit(1)
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    print(f"  Train: {G}{len(tr)}{X} rows | Test: {G}{len(te)}{X} rows")
    return tr, te

def engineer(df):
    df = df.copy()
    df["speed_squared"]     = df["speed_kmh"] ** 2
    df["speed_x_collision"] = df["speed_kmh"] * df["collision_enc"]
    df["age_x_speed"]       = df["vehicle_age_years"] * df["speed_kmh"]
    return df

def prepare(tr, te):
    sec("Feature engineering")
    tr = engineer(tr); te = engineer(te)
    feats = [f for f in FEATURE_COLS if f in tr.columns]
    Xtr = tr[feats].fillna(0); ytr = tr["repair_cost_inr"]
    Xte = te[feats].fillna(0); yte = te["repair_cost_inr"]
    sc  = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=feats)
    Xte_s = pd.DataFrame(sc.transform(Xte),     columns=feats)
    print(f"  Features used: {G}{len(feats)}{X} → {feats}")
    return Xtr, Xte, Xtr_s, Xte_s, ytr, yte, sc, feats

def evaluate(mdl, Xtr, Xte, ytr, yte, key, name):
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    yp = mdl.predict(Xte)
    r2   = r2_score(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    mae  = mean_absolute_error(yte, yp)
    cv   = cross_val_score(mdl, Xtr, ytr, cv=KFold(5,shuffle=True,random_state=42), scoring="r2")
    return dict(key=key,name=name,model=mdl,y_pred=yp,
                r2=round(r2,4), rmse=round(rmse,0), mae=round(mae,0),
                cv_mean=round(cv.mean(),4), cv_std=round(cv.std(),4),
                time_s=round(time.time()-t0,2))

def print_res(r):
    col = G if r["r2"]>=0.85 else (Y if r["r2"]>=0.70 else R)
    print(f"  {B}{r['name']:<22}{X}  R²={col}{r['r2']:.4f}{X}"
          f"  RMSE=₹{r['rmse']:>10,.0f}  MAE=₹{r['mae']:>10,.0f}"
          f"  CV={r['cv_mean']:.3f}±{r['cv_std']:.3f}  ({r['time_s']}s)")

def show_importance(mdl, key, feats, Xte, yte):
    sec("Feature importance")
    if hasattr(mdl,"feature_importances_"):
        imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
    elif hasattr(mdl,"coef_"):
        imp = pd.Series(np.abs(mdl.coef_), index=feats).sort_values(ascending=False)
    else:
        return
    for f, v in imp.items():
        bar = "█" * int(v * 50 / imp.max())
        print(f"  {f:<26} {G}{bar}{X} {v:.4f}")

def save_all(best, sc, feats):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(sc,            f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feats,         f"{MODEL_DIR}/features.pkl")
    meta = {k:v for k,v in best.items() if k not in ("model","y_pred")}
    json.dump(meta, open(f"{MODEL_DIR}/report.json","w"), indent=2)
    sec("Saved")
    print(f"  {G}✓{X} models/best_model.pkl ({best['name']})")
    print(f"  {G}✓{X} models/scaler.pkl  |  models/features.pkl  |  models/report.json")

def interactive_predict(mdl=None, sc=None, feats=None):
    sec("Sample prediction")
    if mdl is None:
        mdl   = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        sc    = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        feats = joblib.load(f"{MODEL_DIR}/features.pkl")

    sample = dict(
        speed_kmh=65, vehicle_age_years=5, vehicle_value_inr=800000,
        occupants=2, airbag_deployed=1, seatbelt_worn=1,
        previous_accidents=1, driver_age=32,
        collision_enc=1.8, road_enc=1.3, weather_enc=1.25,
        vehicle_type_enc=1.0, tyre_enc=1.0,
        speed_squared=65**2, speed_x_collision=65*1.8, age_x_speed=5*65,
    )
    row = pd.DataFrame([[sample.get(f,0) for f in feats]], columns=feats)
    row_s = pd.DataFrame(sc.transform(row), columns=feats)
    pred  = mdl.predict(row_s)[0]

    print(f"\n  Speed          : 65 km/h")
    print(f"  Collision type : Head-on")
    print(f"  Vehicle value  : ₹8,00,000")
    print(f"  Road condition : Wet")
    print(f"\n  {B}{G}Predicted repair cost : ₹{pred:,.0f}{X}")
    sev = "Total Loss" if pred>1500000 else ("Critical" if pred>800000 else
          ("Severe" if pred>400000 else ("Moderate" if pred>150000 else "Minor")))
    print(f"  Damage severity        : {sev}")

def compare_all(results):
    sec("Model comparison")
    print(f"  {'Model':<22} {'R²':>7} {'RMSE (₹)':>14} {'MAE (₹)':>13} {'CV R²':>10} {'Time':>6}")
    print("  " + "─"*72)
    for r in sorted(results, key=lambda x: x["r2"], reverse=True):
        flag = f"{G}◀ best{X}" if r == max(results,key=lambda x:x["r2"]) else ""
        print(f"  {r['name']:<22} {r['r2']:>7.4f} {r['rmse']:>14,.0f} {r['mae']:>13,.0f}"
              f" {r['cv_mean']:>7.3f}±{r['cv_std']:.3f} {r['time_s']:>5.2f}s  {flag}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="all")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    banner("🚗  Car Repair Cost ML Predictor — Training Pipeline", G)

    if args.predict and args.model=="all" and not args.compare:
        interactive_predict(); return

    tr, te = load()
    Xtr, Xte, Xtr_s, Xte_s, ytr, yte, sc, feats = prepare(tr, te)

    keys = list(MODEL_REGISTRY.keys()) if (args.model=="all" or args.compare) else [args.model]
    sec(f"Training {len(keys)} model(s)")

    results = []
    for k in keys:
        name, mdl = MODEL_REGISTRY[k]
        xtr = Xtr_s if k in LINEAR else Xtr
        xte = Xte_s if k in LINEAR else Xte
        res = evaluate(mdl, xtr, xte, ytr, yte, k, name)
        print_res(res); results.append(res)

    if len(results) > 1 or args.compare:
        compare_all(results)

    best = max(results, key=lambda x: x["r2"])
    banner(f"Best: {best['name']}  R²={best['r2']}  RMSE=₹{best['rmse']:,.0f}", G)

    xte_imp = Xte_s if best["key"] in LINEAR else Xte
    show_importance(best["model"], best["key"], feats, xte_imp, yte)

    if not args.no_save:
        save_all(best, sc, feats)

    if args.predict:
        interactive_predict(best["model"], sc, feats)

    banner("Training complete ✓", G)

if __name__ == "__main__":
    main()
