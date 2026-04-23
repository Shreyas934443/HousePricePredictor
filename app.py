# =============================================================================
# 🏠 Melbourne House Price Predictor — Streamlit App
# Step 7: Deployment
#
# HOW TO RUN:
#   1. Copy this file into your project folder:
#         D:\House Price Predictor\
#      alongside model.pkl, scaler.pkl, features.pkl
#
#   2. Install streamlit (one time only):
#         pip install streamlit
#
#   3. In VS Code terminal, run:
#         streamlit run app.py
#
#   4. A browser tab opens automatically at http://localhost:8501
#
# YOUR MODEL'S 9 SELECTED FEATURES (from features.pkl):
#   Rooms, Type_enc, DistanceSquared_log, DistanceSquared,
#   Distance, TotalRooms, IsNewProperty, IsCloseToCBD, HasParking
# =============================================================================

import streamlit as st
import joblib
import numpy as np

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp                      { background-color: #0F1923; }

section[data-testid="stSidebar"] {
    background-color: #131F2E;
    border-right: 1px solid #1E3148;
}
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1D4ED8, #0EA5E9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────────────────────
# @st.cache_resource loads the files once and reuses them across all button
# clicks — without this, the model reloads every time which is very slow
@st.cache_resource
def load_assets():
    model    = joblib.load("model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

try:
    model, scaler, features = load_assets()
    model_ok = True
except FileNotFoundError as e:
    model_ok    = False
    load_error  = str(e)


# ── Feature builder ───────────────────────────────────────────────────────────
def build_input_vector(rooms, distance, prop_type, bathroom, car,
                       year_built, features_list):
    """
    Maps user inputs to the exact 9-feature vector the model expects.

    Your notebook used LabelEncoder on Type in alphabetical order:
        h → 0,  t → 1,  u → 2
    One-hot encoding with drop_first=True removed 'h', keeping 't' and 'u'.

    We fill all 9 features here. Any key not in features_list is ignored
    by the list comprehension at the end.
    """
    sale_year = 2018                          # median year in training data
    prop_age  = max(0, sale_year - year_built)

    # LabelEncoder assigns alphabetical order: h=0, t=1, u=2
    type_enc  = {'h': 0, 't': 1, 'u': 2}.get(prop_type, 0)

    derived = {
        # Core numeric
        'Rooms':               rooms,
        'Distance':            distance,
        'Bathroom':            bathroom,
        'Car':                 car,
        'Propertycount':       5000,
        'SaleYear':            sale_year,
        'SaleMonth':           6,
        # Engineered
        'PropertyAge':         prop_age,
        'TotalRooms':          rooms + bathroom,
        'RoomToBathroomRatio': rooms / max(bathroom, 1),
        'IsNewProperty':       int(prop_age <= 10),
        'IsCloseToCBD':        int(distance <= 10),
        'HasParking':          int(car > 0),
        'DistanceSquared':     distance ** 2,
        'DistanceSquared_log': np.log1p(distance ** 2),
        'LandsizePerRoom_log': np.log1p(500 / max(rooms, 1)),
        'Landsize_log':        np.log1p(500),
        # Categorical encodings
        'Type_enc':            type_enc,
        'Suburb_enc':          50,
        'Regionname_enc':      1,
        'CouncilArea_enc':     5,
        'SellerG_enc':         3,
        'Method_enc':          4,
        # One-hot Type (drop_first drops 'h', keeps 't' and 'u')
        'Type_t':              int(prop_type == 't'),
        'Type_u':              int(prop_type == 'u'),
        'Type_t_log':          np.log1p(int(prop_type == 't')),
        'Type_u_log':          np.log1p(int(prop_type == 'u')),
        # One-hot Method (default to most common: Sold)
        'Method_SA':           0,
        'Method_SP':           0,
        'Method_SA_log':       0.0,
        'Method_SP_log':       0.0,
    }

    # Build vector in EXACT order from features.pkl
    vector = np.array([derived.get(f, 0) for f in features_list], dtype=float)
    return vector.reshape(1, -1)


# ── Confidence interval ───────────────────────────────────────────────────────
def confidence_interval(price, mae=149537):
    """Returns low/high range using ±MAE from Step 6 evaluation."""
    return max(0, price - mae), price + mae


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:

    st.markdown("""
    <div style='padding-bottom:1rem;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.45rem;color:#F1F5F9;'>
            🏠 Property Details
        </div>
        <div style='color:#64748B;font-size:0.8rem;margin-top:0.3rem;'>
            Adjust sliders to match your property
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #1E3148;margin-bottom:1rem;'>
    """, unsafe_allow_html=True)

    # Property type
    st.markdown("**Property Type**")
    prop_type = st.radio(
        "prop_type", ["h", "u", "t"],
        format_func=lambda x: {
            "h": "🏡  House / Villa / Cottage",
            "u": "🏢  Unit / Apartment",
            "t": "🏘  Townhouse"
        }[x],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1E3148;margin:0.8rem 0;'>",
                unsafe_allow_html=True)

    # Rooms
    st.markdown("**Rooms & Spaces**")
    rooms    = st.slider("Bedrooms",   1, 6, 3)
    bathroom = st.slider("Bathrooms",  1, 4, 2)
    car      = st.slider("Car Spaces", 0, 4, 1)

    st.markdown("<hr style='border:none;border-top:1px solid #1E3148;margin:0.8rem 0;'>",
                unsafe_allow_html=True)

    # Location
    st.markdown("**Location**")
    distance = st.slider("Distance from CBD (km)", 1.0, 45.0, 10.0, step=0.5)

    st.markdown("<hr style='border:none;border-top:1px solid #1E3148;margin:0.8rem 0;'>",
                unsafe_allow_html=True)

    # Year built
    st.markdown("**Property Age**")
    year_built = st.number_input("Year Built", 1850, 2023, 1990, step=1)
    prop_age   = max(0, 2018 - year_built)
    st.markdown(
        f"<div style='color:#64748B;font-size:0.78rem;margin-top:-0.4rem;'>"
        f"Age at sale: {prop_age} years</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Predict Price", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Model info footer
    st.markdown("""
    <div style='background:#0F1923;border:1px solid #1E3148;border-radius:8px;
                padding:0.85rem 1rem;font-size:0.78rem;'>
        <div style='color:#64748B;text-transform:uppercase;letter-spacing:0.06em;
                    font-weight:500;margin-bottom:0.55rem;'>Model Info</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
            <span style='color:#64748B;'>Algorithm</span>
            <span style='color:#38BDF8;font-weight:500;'>Gradient Boosting</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
            <span style='color:#64748B;'>R² Score</span>
            <span style='color:#F1F5F9;'>0.837</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
            <span style='color:#64748B;'>MAE</span>
            <span style='color:#F1F5F9;'>~$149K</span>
        </div>
        <div style='display:flex;justify-content:space-between;'>
            <span style='color:#64748B;'>Training rows</span>
            <span style='color:#F1F5F9;'>13,580</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN PANEL
# =============================================================================

# Title
st.markdown("""
<div style='padding-bottom:0.4rem;'>
    <div style='font-family:DM Serif Display,serif;font-size:2.6rem;
                color:#F1F5F9;line-height:1.1;'>
        Melbourne House<br>Price Predictor
    </div>
    <div style='color:#64748B;font-size:0.92rem;font-weight:300;margin-top:0.4rem;'>
        Gradient Boosting &nbsp;·&nbsp; 13,580 Melbourne properties
        &nbsp;·&nbsp; R² = 0.837 &nbsp;·&nbsp; MAE ≈ $149K
    </div>
</div>
<hr style='border:none;border-top:1px solid #1E3148;margin:1rem 0 1.4rem 0;'>
""", unsafe_allow_html=True)

# File not found error
if not model_ok:
    st.error(f"""
    ⚠️ **Model files not found.**

    Make sure all three files are in the **same folder** as `app.py`:
    - `model.pkl`
    - `scaler.pkl`
    - `features.pkl`

    Then restart with `streamlit run app.py`

    Error: `{load_error}`
    """)
    st.stop()

# Two columns
col_left, col_right = st.columns([1.3, 1], gap="large")

# ── LEFT column — result ──────────────────────────────────────────────────────
with col_left:

    if predict_clicked:

        # Build and scale input
        X_raw    = build_input_vector(rooms, distance, prop_type,
                                      bathroom, car, year_built, features)
        X_scaled = scaler.transform(X_raw)
        log_pred = model.predict(X_scaled)[0]
        price    = np.expm1(log_pred)
        low, high = confidence_interval(price)

        # ── Price result card ──────────────────────────────────────────────
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E3A5F 0%,#131F2E 100%);
                    border:1px solid #2563EB55;border-radius:16px;
                    padding:2rem 2.5rem;text-align:center;margin-bottom:1.2rem;'>
            <div style='color:#64748B;font-size:0.75rem;text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:500;margin-bottom:0.5rem;'>
                Estimated Property Value
            </div>
            <div style='font-family:DM Serif Display,serif;font-size:3.2rem;
                        color:#38BDF8;line-height:1;margin-bottom:0.55rem;'>
                ${price:,.0f}
            </div>
            <div style='color:#64748B;font-size:0.85rem;'>
                Range &nbsp;
                <span style='color:#94A3B8;font-weight:500;'>${low:,.0f}</span>
                &nbsp;—&nbsp;
                <span style='color:#94A3B8;font-weight:500;'>${high:,.0f}</span>
                &nbsp; (± MAE)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Quick metrics ──────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        metrics = [
            (m1, "Per Bedroom",  f"${price/max(rooms,1):,.0f}"),
            (m2, "Total Rooms",  str(rooms + bathroom)),
            (m3, "From CBD",     f"{distance} km"),
        ]
        for col_m, label, val in metrics:
            col_m.markdown(f"""
            <div style='background:#131F2E;border:1px solid #1E3148;border-radius:10px;
                        padding:0.85rem;text-align:center;'>
                <div style='font-size:1.2rem;font-weight:600;color:#F1F5F9;'>{val}</div>
                <div style='font-size:0.7rem;color:#64748B;text-transform:uppercase;
                            letter-spacing:0.06em;margin-top:0.2rem;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Insights ───────────────────────────────────────────────────────
        insights = []
        if distance <= 10:
            insights.append("📍 <b>Inner city location</b> — strong CBD proximity premium.")
        elif distance <= 20:
            insights.append("📍 <b>Mid-ring suburb</b> — balance of price and accessibility.")
        else:
            insights.append("📍 <b>Outer suburb</b> — more land, lower price density.")

        if prop_age <= 10:
            insights.append("🏗 <b>New build</b> — modern construction premium factored in.")
        elif prop_age >= 60:
            insights.append("🏚 <b>Heritage property</b> — renovation potential not modelled.")

        if car == 0 and distance <= 10:
            insights.append("🚗 <b>No parking</b> in inner suburb — limits buyer pool.")
        if rooms >= 4:
            insights.append("🛏 <b>Large property</b> — family size adds significant premium.")
        if prop_type == 'h':
            insights.append("🏡 <b>House</b> — commands the highest median price of all types.")

        for ins in insights:
            st.markdown(f"""
            <div style='background:#131F2E;border-left:3px solid #38BDF8;
                        border-radius:0 8px 8px 0;padding:0.55rem 1rem;
                        color:#94A3B8;font-size:0.84rem;margin-bottom:0.45rem;'>
                {ins}
            </div>
            """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div style='color:#374151;font-size:0.73rem;margin-top:1rem;line-height:1.5;'>
            ℹ️ Estimate based on a Gradient Boosting model (R²=0.837, MAE≈$149K)
            trained on Melbourne sales data. Not a formal property valuation.
            Consult a licensed real estate agent for professional advice.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Default waiting state
        st.markdown("""
        <div style='background:#131F2E;border:1px dashed #1E3148;border-radius:16px;
                    padding:4rem 2rem;text-align:center;margin-top:0.5rem;'>
            <div style='font-size:3.5rem;margin-bottom:1rem;'>🏠</div>
            <div style='font-family:DM Serif Display,serif;font-size:1.4rem;
                        color:#F1F5F9;margin-bottom:0.6rem;'>
                Ready to Predict
            </div>
            <div style='color:#64748B;font-size:0.88rem;line-height:1.8;'>
                Set the property details in the sidebar<br>
                then click
                <span style='color:#38BDF8;font-weight:600;'>Predict Price</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT column — summary & importance ───────────────────────────────────────
with col_right:

    # Property summary table
    st.markdown("""
    <div style='color:#64748B;font-size:0.73rem;text-transform:uppercase;
                letter-spacing:0.1em;font-weight:500;margin-bottom:0.9rem;'>
        Property Summary
    </div>
    """, unsafe_allow_html=True)

    type_lbl  = {'h': '🏡 House', 'u': '🏢 Unit/Apt', 't': '🏘 Townhouse'}
    new_badge = (" <span style='background:#14532D;color:#4ADE80;font-size:0.65rem;"
                 "border-radius:4px;padding:0.1rem 0.35rem;'>NEW</span>"
                 if prop_age <= 10 else "")
    cbd_badge = (" <span style='background:#1E3A5F;color:#38BDF8;font-size:0.65rem;"
                 "border-radius:4px;padding:0.1rem 0.35rem;'>INNER</span>"
                 if distance <= 10 else "")

    summary_rows = [
        ("Type",         type_lbl[prop_type]),
        ("Bedrooms",     str(rooms)),
        ("Bathrooms",    str(bathroom)),
        ("Car spaces",   str(car)),
        ("Distance",     f"{distance} km{cbd_badge}"),
        ("Year built",   f"{year_built}{new_badge}"),
        ("Property age", f"{prop_age} yrs"),
        ("Total rooms",  str(rooms + bathroom)),
        ("Is new",       "Yes ✅" if prop_age <= 10 else "No"),
        ("Has parking",  "Yes ✅" if car > 0 else "No"),
    ]

    for label, value in summary_rows:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:0.5rem 0;border-bottom:1px solid #1E3148;'>
            <span style='color:#64748B;font-size:0.8rem;'>{label}</span>
            <span style='color:#F1F5F9;font-size:0.84rem;font-weight:500;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance chart (from your RF selector results)
    st.markdown("""
    <div style='color:#64748B;font-size:0.73rem;text-transform:uppercase;
                letter-spacing:0.1em;font-weight:500;margin-bottom:0.9rem;'>
        What Drives Price
    </div>
    """, unsafe_allow_html=True)

    # These are the exact importance values from your notebook's RF selector
    top_features = [
        ("Rooms",              38.4),
        ("Property Type",      19.1),
        ("Distance² (log)",    13.8),
        ("Distance²",          11.9),
        ("Distance",           10.8),
        ("Total Rooms",         1.1),
    ]
    max_imp = top_features[0][1]

    for feat_name, imp in top_features:
        bar_w = int((imp / max_imp) * 100)
        st.markdown(f"""
        <div style='margin-bottom:0.6rem;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:0.18rem;'>
                <span style='color:#94A3B8;font-size:0.77rem;'>{feat_name}</span>
                <span style='color:#64748B;font-size:0.73rem;'>{imp:.1f}%</span>
            </div>
            <div style='background:#1E3148;border-radius:4px;height:5px;'>
                <div style='background:linear-gradient(90deg,#1D4ED8,#38BDF8);
                            width:{bar_w}%;height:5px;border-radius:4px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tips box
    st.markdown("""
    <div style='background:#131F2E;border:1px solid #1E3148;border-radius:10px;
                padding:1rem 1.1rem;'>
        <div style='color:#64748B;font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.06em;font-weight:500;margin-bottom:0.6rem;'>
            Tips to Increase Value
        </div>
        <div style='color:#94A3B8;font-size:0.79rem;line-height:1.8;'>
            🛏 &nbsp;Each extra bedroom is the biggest price driver<br>
            📍 &nbsp;Closer to CBD significantly raises price<br>
            🏡 &nbsp;Houses command highest median vs units/townhouses<br>
            🏗 &nbsp;New builds attract a premium over older properties<br>
            🚗 &nbsp;Parking is valuable especially under 10km from CBD
        </div>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#1E3148;font-size:0.7rem;padding-bottom:1rem;'>
    Melbourne Housing Dataset &nbsp;·&nbsp; Gradient Boosting &nbsp;·&nbsp;
    scikit-learn &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)