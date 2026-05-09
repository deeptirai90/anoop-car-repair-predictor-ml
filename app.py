"""
app.py — Car Repair Cost Predictor Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import glob
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Repair Cost Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #FF6B35, #F7C59F, #EFEFD0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.hero-sub { color: #888; font-size: 1rem; margin-bottom: 2rem; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 14px; padding: 1.2rem 1rem;
    text-align: center; margin-bottom: 0.5rem;
}
.metric-card .val { font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700; color:#FF6B35; }
.metric-card .lbl { font-size:0.78rem; color:#888; text-transform:uppercase; letter-spacing:.08em; }

.severity-minor    { background:#1a3a1a; border:1px solid #4CAF50; border-radius:8px; padding:8px 14px; color:#81C784; font-weight:600; }
.severity-moderate { background:#3a2a0a; border:1px solid #FF9800; border-radius:8px; padding:8px 14px; color:#FFB74D; font-weight:600; }
.severity-severe   { background:#3a1a0a; border:1px solid #FF5722; border-radius:8px; padding:8px 14px; color:#FF8A65; font-weight:600; }
.severity-critical { background:#2a0a0a; border:1px solid #f44336; border-radius:8px; padding:8px 14px; color:#EF9A9A; font-weight:600; }
.severity-total    { background:#1a0a0a; border:1px solid #880000; border-radius:8px; padding:8px 14px; color:#CF6679; font-weight:600; }

.result-box {
    background: linear-gradient(135deg, #0d1f0d, #1a3a1a);
    border: 2px solid #4CAF50; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
}
.result-price { font-family:'Rajdhani',sans-serif; font-size:3.2rem; font-weight:700; color:#66BB6A; }
.result-range { font-size:0.9rem; color:#aaa; margin-top:0.4rem; }

.section-hdr {
    font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:600;
    color:#FF6B35; border-left:4px solid #FF6B35;
    padding-left:12px; margin:1.5rem 0 1rem;
}
.img-caption {
    font-size:0.78rem; color:#888; text-align:center; margin-top:4px;
}
</style>
""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
    train_path = "data/car_incidents_train.csv"
    test_path  = "data/car_incidents_test.csv"
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)
    # Generate on the fly
    np.random.seed(42); n = 1000
    speed           = np.random.randint(5, 141, n)
    vehicle_age     = np.random.randint(0, 21, n)
    vehicle_value   = np.random.randint(300000, 3000001, n)
    occupants       = np.random.randint(1, 6, n)
    airbag_deployed = np.clip((speed > 40).astype(int) + np.random.binomial(1,0.1,n), 0, 1)
    collision_type  = np.random.choice(["Head-on","Rear-end","Side","Rollover","Pole"], n, p=[.25,.30,.25,.10,.10])
    road_condition  = np.random.choice(["Dry","Wet","Icy","Gravel"], n, p=[.50,.30,.10,.10])
    weather         = np.random.choice(["Clear","Rain","Fog","Snow"], n, p=[.55,.25,.12,.08])
    vehicle_type    = np.random.choice(["Sedan","SUV","Hatchback","Truck","Two-wheeler"], n, p=[.35,.25,.20,.10,.10])
    insurance_type  = np.random.choice(["Comprehensive","Third-party","None"], n, p=[.55,.35,.10])
    seatbelt_worn   = np.random.binomial(1, 0.75, n)
    prev_acc        = np.random.randint(0, 5, n)
    driver_age      = np.random.randint(18, 71, n)
    tyre_condition  = np.random.choice(["Good","Worn","Bald"], n, p=[.60,.28,.12])
    ce = np.array([{"Head-on":1.8,"Rear-end":1.0,"Side":1.3,"Rollover":2.2,"Pole":1.5}[c] for c in collision_type])
    re = np.array([{"Dry":1.0,"Wet":1.3,"Icy":1.8,"Gravel":1.2}[r] for r in road_condition])
    we = np.array([{"Clear":1.0,"Rain":1.25,"Fog":1.15,"Snow":1.4}[w] for w in weather])
    ve = np.array([{"Sedan":1.0,"SUV":1.3,"Hatchback":0.85,"Truck":1.5,"Two-wheeler":0.6}[v] for v in vehicle_type])
    te = np.array([{"Good":1.0,"Worn":1.15,"Bald":1.35}[t] for t in tyre_condition])
    noise = np.random.normal(0, 8000, n)
    cost = np.clip((speed**2*18 + vehicle_value*0.08 + vehicle_age*2500 + airbag_deployed*45000
                    + prev_acc*5000 - seatbelt_worn*8000 + (70-driver_age.clip(18,70))*400 + noise
                    ) * ce*re*we*ve*te, 5000, None).round(-2).astype(int)
    def sev(s): return "Minor" if s<30 else "Moderate" if s<60 else "Severe" if s<90 else "Critical" if s<110 else "Total Loss"
    df = pd.DataFrame({"speed_kmh":speed,"vehicle_age_years":vehicle_age,"vehicle_value_inr":vehicle_value,
                       "occupants":occupants,"airbag_deployed":airbag_deployed,"collision_type":collision_type,
                       "road_condition":road_condition,"weather":weather,"vehicle_type":vehicle_type,
                       "insurance_type":insurance_type,"seatbelt_worn":seatbelt_worn,"previous_accidents":prev_acc,
                       "driver_age":driver_age,"tyre_condition":tyre_condition,
                       "collision_enc":ce.round(2),"road_enc":re.round(2),"weather_enc":we.round(2),
                       "vehicle_type_enc":ve.round(2),"tyre_enc":te.round(2),
                       "damage_severity":[sev(s) for s in speed],"repair_cost_inr":cost})
    train = df.sample(frac=0.8, random_state=42)
    test  = df.drop(train.index)
    return train, test

FEATURE_COLS = ["speed_kmh","vehicle_age_years","vehicle_value_inr","occupants",
                "airbag_deployed","seatbelt_worn","previous_accidents","driver_age",
                "collision_enc","road_enc","weather_enc","vehicle_type_enc","tyre_enc",
                "speed_squared","speed_x_collision","age_x_speed"]

def engineer(df):
    df = df.copy()
    df["speed_squared"]     = df["speed_kmh"]**2
    df["speed_x_collision"] = df["speed_kmh"]*df["collision_enc"]
    df["age_x_speed"]       = df["vehicle_age_years"]*df["speed_kmh"]
    return df

@st.cache_resource
def train_model(train_df, test_df):
    train_df = engineer(train_df); test_df = engineer(test_df)
    feats = [f for f in FEATURE_COLS if f in train_df.columns]
    Xtr = train_df[feats]; ytr = train_df["repair_cost_inr"]
    Xte = test_df[feats];  yte = test_df["repair_cost_inr"]
    sc  = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    mdl = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.06,
                                     subsample=0.8, random_state=42)
    mdl.fit(Xtr, ytr)
    yp   = mdl.predict(Xte)
    r2   = r2_score(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    mae  = mean_absolute_error(yte, yp)
    cv   = cross_val_score(mdl, Xtr, ytr, cv=5, scoring="r2")
    return mdl, sc, feats, Xtr, Xte, ytr, yte, yp, r2, rmse, mae, cv

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_severity(cost):
    if cost < 50000:   return "Minor",     "#4CAF50", "severity-minor"
    if cost < 200000:  return "Moderate",  "#FF9800", "severity-moderate"
    if cost < 500000:  return "Severe",    "#FF5722", "severity-severe"
    if cost < 1000000: return "Critical",  "#f44336", "severity-critical"
    return "Total Loss", "#880000", "severity-total"

def encode(val, mapping): return mapping.get(val, 1.0)

COLL_MAP  = {"Head-on":1.8,"Rear-end":1.0,"Side":1.3,"Rollover":2.2,"Pole":1.5}
ROAD_MAP  = {"Dry":1.0,"Wet":1.3,"Icy":1.8,"Gravel":1.2}
WEAT_MAP  = {"Clear":1.0,"Rain":1.25,"Fog":1.15,"Snow":1.4}
VTYP_MAP  = {"Sedan":1.0,"SUV":1.3,"Hatchback":0.85,"Truck":1.5,"Two-wheeler":0.6}
TYRE_MAP  = {"Good":1.0,"Worn":1.15,"Bald":1.35}

# ── Load data & train ─────────────────────────────────────────────────────────
train_df, test_df = load_or_generate_data()
mdl, sc, feats, Xtr, Xte, ytr, yte, yp, r2, rmse, mae, cv = train_model(train_df, test_df)

# ══ HEADER ════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🚗 Car Repair Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered repair cost estimation from accident speed & incident parameters · Automotive ML</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown(f"**Model:** Gradient Boosting")
    st.markdown(f"**R² Score:** `{r2:.4f}`")
    st.markdown(f"**RMSE:** `₹{rmse:,.0f}`")
    st.markdown(f"**Training samples:** `{len(train_df)}`")
    st.markdown("---")
    st.markdown("**Quick Speed Scenario**")
    quick_speed = st.select_slider("Speed (km/h)", options=[15,25,35,45,55,65,75,85,100,120], value=65)
    st.markdown("---")
    st.caption("Automotive ML · Streamlit + Scikit-learn")

# ══ TABS ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard", "🔮 Predict Cost", "🖼️ Incident Gallery",
    "📈 Model Analysis", "📋 Dataset"
])

# ─── TAB 1: Dashboard ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-hdr">Model Performance</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("R² Score",         f"{r2:.4f}")
    c2.metric("RMSE",             f"₹{rmse:,.0f}")
    c3.metric("MAE",              f"₹{mae:,.0f}")
    c4.metric("CV R² (5-fold)",   f"{cv.mean():.3f} ± {cv.std():.3f}")
    c5.metric("Training samples", len(train_df))

    st.markdown('<div class="section-hdr">Data Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.hist(train_df["repair_cost_inr"]/100000, bins=35, color="#FF6B35", edgecolor="#0e1117", alpha=0.85)
        ax.set_xlabel("Repair Cost (₹ Lakhs)", color="#ccc"); ax.set_ylabel("Count", color="#ccc")
        ax.tick_params(colors="#888"); ax.set_title("Repair Cost Distribution", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        sev_counts = train_df["damage_severity"].value_counts()
        colors = ["#4CAF50","#FF9800","#FF5722","#f44336","#880000"]
        bars = ax.bar(sev_counts.index, sev_counts.values, color=colors[:len(sev_counts)], edgecolor="#0e1117")
        ax.set_xlabel("Severity", color="#ccc"); ax.set_ylabel("Count", color="#ccc")
        ax.tick_params(colors="#888", axis='x', rotation=15)
        ax.set_title("Incident Severity Distribution", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.scatter(train_df["speed_kmh"], train_df["repair_cost_inr"]/100000,
                   alpha=0.3, color="#FF6B35", s=8)
        ax.set_xlabel("Speed (km/h)", color="#ccc"); ax.set_ylabel("Cost (₹ L)", color="#ccc")
        ax.tick_params(colors="#888"); ax.set_title("Speed vs Repair Cost", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        coll_avg = train_df.groupby("collision_type")["repair_cost_inr"].mean().sort_values() / 100000
        ax.barh(coll_avg.index, coll_avg.values, color="#FF6B35", edgecolor="#0e1117")
        ax.set_xlabel("Avg Cost (₹ L)", color="#ccc")
        ax.tick_params(colors="#888"); ax.set_title("Avg Cost by Collision Type", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 2: Predict ───────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-hdr">Enter Incident Details</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            speed         = st.slider("🚀 Speed at Impact (km/h)", 5, 140, quick_speed, 5)
            vehicle_age   = st.slider("📅 Vehicle Age (years)",     0, 20,  5, 1)
            vehicle_value = st.number_input("💰 Vehicle Value (₹)", 100000, 5000000, 800000, 50000)
        with r1c2:
            collision_type = st.selectbox("💥 Collision Type", list(COLL_MAP.keys()))
            road_condition = st.selectbox("🛣️ Road Condition",  list(ROAD_MAP.keys()))
            weather        = st.selectbox("🌦️ Weather",         list(WEAT_MAP.keys()))
        with r1c3:
            vehicle_type   = st.selectbox("🚗 Vehicle Type",    list(VTYP_MAP.keys()))
            tyre_condition = st.selectbox("🔵 Tyre Condition",  list(TYRE_MAP.keys()))
            occupants      = st.slider("👥 Occupants",          1, 5, 2, 1)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            airbag_deployed    = st.selectbox("🛡️ Airbag Deployed",  ["Yes","No"])
            seatbelt_worn      = st.selectbox("🔒 Seatbelt Worn",    ["Yes","No"])
        with r2c2:
            driver_age         = st.slider("👤 Driver Age",     18, 70, 32, 1)
            previous_accidents = st.slider("⚠️ Previous Accidents", 0, 4, 0, 1)
        with r2c3:
            insurance_type     = st.selectbox("📋 Insurance",   ["Comprehensive","Third-party","None"])

        submitted = st.form_submit_button("🔮 Predict Repair Cost", use_container_width=True)

    if submitted:
        ce = encode(collision_type, COLL_MAP)
        re = encode(road_condition, ROAD_MAP)
        we = encode(weather,        WEAT_MAP)
        ve = encode(vehicle_type,   VTYP_MAP)
        te = encode(tyre_condition, TYRE_MAP)
        ab = 1 if airbag_deployed=="Yes" else 0
        sb = 1 if seatbelt_worn=="Yes"   else 0

        row = pd.DataFrame([[
            speed, vehicle_age, vehicle_value, occupants, ab, sb,
            previous_accidents, driver_age, ce, re, we, ve, te,
            speed**2, speed*ce, vehicle_age*speed
        ]], columns=feats)

        pred  = mdl.predict(row)[0]
        lo    = max(0, pred - rmse)
        hi    = pred + rmse
        sev, scolor, scls = get_severity(pred)

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:1rem;color:#aaa;margin-bottom:.4rem">Estimated Repair Cost</div>
            <div class="result-price">₹ {pred:,.0f}</div>
            <div class="result-range">Confidence range: ₹{lo:,.0f} — ₹{hi:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity",     sev)
        m2.metric("Speed",        f"{speed} km/h")
        m3.metric("Collision",    collision_type)
        m4.metric("Insurance",    insurance_type)

        # Parts likely damaged
        st.markdown("**🔧 Likely Damaged Parts:**")
        parts = []
        if speed > 10:  parts.append("🔵 Bumper / Body Panel")
        if speed > 25:  parts.append("🔵 Hood / Bonnet")
        if speed > 40:  parts.append("🟡 Headlights / Taillights")
        if speed > 55:  parts.append("🟡 Radiator / Cooling System")
        if speed > 70:  parts.append("🔴 Airbags (deployed)" if ab else "🔴 Frame / Chassis")
        if speed > 90:  parts.append("🔴 Engine / Transmission")
        if speed > 110: parts.append("🚨 Total Vehicle Loss")
        for p in parts:
            st.write(f"  {p}")

        # Show matching incident image
        img_files = sorted(glob.glob("data/images/incident_*.png"))
        if img_files:
            speeds_avail = [int(f.split("_speed_")[1].replace("kmh.png","")) for f in img_files]
            closest = min(speeds_avail, key=lambda s: abs(s-speed))
            match_img = [f for f in img_files if f"speed_{closest}kmh" in f]
            if match_img:
                st.markdown(f"**📸 Closest matching incident image (speed: {closest} km/h):**")
                st.image(match_img[0], use_container_width=True)

# ─── TAB 3: Incident Gallery ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-hdr">10 Incident Reference Images</div>', unsafe_allow_html=True)
    st.caption("Synthetic incident visualizations showing damage severity at different impact speeds.")

    img_files = sorted(glob.glob("data/images/incident_*.png"))

    INCIDENT_META = [
        {"speed":15,  "damage":"Minor",    "cost":"₹12,000",  "parts":"Front Bumper"},
        {"speed":25,  "damage":"Minor",    "cost":"₹28,000",  "parts":"Hood, Headlight"},
        {"speed":35,  "damage":"Moderate", "cost":"₹55,000",  "parts":"Fender, Door, Bumper"},
        {"speed":45,  "damage":"Moderate", "cost":"₹85,000",  "parts":"Hood, Radiator, Airbag"},
        {"speed":55,  "damage":"Severe",   "cost":"₹1,30,000","parts":"Engine Bay, Frame"},
        {"speed":65,  "damage":"Severe",   "cost":"₹1,85,000","parts":"Full Front, Chassis"},
        {"speed":75,  "damage":"Critical", "cost":"₹2,45,000","parts":"Total Front, Engine"},
        {"speed":85,  "damage":"Critical", "cost":"₹3,10,000","parts":"Full Body, Roof"},
        {"speed":100, "damage":"Total Loss","cost":"₹4,20,000","parts":"Total Loss"},
        {"speed":120, "damage":"Total Loss","cost":"₹5,80,000","parts":"Total Loss + Liability"},
    ]

    SEV_COLOR = {"Minor":"🟢","Moderate":"🟡","Severe":"🟠","Critical":"🔴","Total Loss":"⚫"}

    if img_files:
        for row_start in range(0, len(img_files), 2):
            cols = st.columns(2)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx >= len(img_files): break
                meta = INCIDENT_META[idx]
                with col:
                    st.image(img_files[idx], use_container_width=True)
                    sev_icon = SEV_COLOR.get(meta["damage"], "⚪")
                    st.markdown(f"""
                    <div style="background:#1a1a2e;border-radius:8px;padding:8px 12px;margin-bottom:1rem;
                                border-left:3px solid #FF6B35;font-size:0.85rem;">
                        <b>#{idx+1} · {meta['speed']} km/h</b> &nbsp;|&nbsp;
                        {sev_icon} {meta['damage']} &nbsp;|&nbsp;
                        💰 {meta['cost']}<br>
                        🔧 {meta['parts']}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Run `python generate_car_images.py` to generate incident images.")
        st.info("Images will appear here once generated.")

# ─── TAB 4: Model Analysis ────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-hdr">Actual vs Predicted</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.scatter(yte/100000, yp/100000, alpha=0.4, color="#FF6B35", s=12)
        mn, mx = (min(yte.min(), yp.min())/100000), (max(yte.max(), yp.max())/100000)
        ax.plot([mn,mx],[mn,mx],"--",color="#4CAF50",linewidth=1.5,label="Perfect fit")
        ax.set_xlabel("Actual (₹ L)", color="#ccc"); ax.set_ylabel("Predicted (₹ L)", color="#ccc")
        ax.tick_params(colors="#888"); ax.legend(fontsize=9)
        ax.set_title("Actual vs Predicted Cost", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        residuals = yte.values - yp
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.scatter(yp/100000, residuals/100000, alpha=0.4, color="#f44336", s=12)
        ax.axhline(0, color="#4CAF50", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Predicted (₹ L)", color="#ccc"); ax.set_ylabel("Residuals (₹ L)", color="#ccc")
        ax.tick_params(colors="#888"); ax.set_title("Residual Plot", color="#fff")
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Feature Importance</div>', unsafe_allow_html=True)
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feats)*0.45)), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    colors = ["#FF6B35" if v > imp.quantile(0.75) else "#FF9800" if v > imp.quantile(0.5) else "#888"
              for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors, edgecolor="#0e1117")
    ax.set_xlabel("Importance", color="#ccc"); ax.tick_params(colors="#888")
    ax.set_title("Feature Importance — Gradient Boosting", color="#fff")
    for sp in ax.spines.values(): sp.set_color("#333")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 5: Dataset ──────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-hdr">Training Dataset</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total rows",         len(train_df)+len(test_df))
    c2.metric("Training rows",      len(train_df))
    c3.metric("Test rows",          len(test_df))
    c4.metric("Mean repair cost",   f"₹{(train_df['repair_cost_inr'].mean()/100000):.1f}L")

    display_cols = ["speed_kmh","vehicle_age_years","vehicle_value_inr","collision_type",
                    "road_condition","weather","vehicle_type","airbag_deployed","damage_severity","repair_cost_inr"]
    avail = [c for c in display_cols if c in train_df.columns]
    st.dataframe(train_df[avail].head(50).style.background_gradient(
        subset=["repair_cost_inr"], cmap="YlOrRd"), use_container_width=True)

    csv = train_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Train CSV", csv, "car_incidents_train.csv", "text/csv")
    csv2 = test_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Test CSV",  csv2,"car_incidents_test.csv",  "text/csv")

    st.markdown('<div class="section-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(train_df[num_cols].describe().round(2), use_container_width=True)
