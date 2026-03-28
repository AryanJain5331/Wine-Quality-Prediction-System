"""
Wine Quality Assessment — GOD-LEVEL EDITION
All immersive features, only prediction section
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Vino Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LUXURY DARK WINE CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Montserrat:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background-color: #0d0608 !important;
    color: #e8ddd0 !important;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #2a0a12 0%, #0d0608 50%, #080308 100%) !important;
}

/* Remove default streamlit padding */
.block-container { padding-top: 2rem !important; max-width: 1400px !important; }

/* ── Hero Header ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.45em;
    color: #c9a96e;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 300;
    line-height: 1.05;
    color: #f5ede0;
    margin: 0 0 0.5rem;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: #c9a96e;
}
.hero-sub {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: #8a7060;
    letter-spacing: 0.15em;
}
.hero-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a96e, transparent);
    margin: 1.5rem auto;
}

/* ── Accuracy Badge ── */
.accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: linear-gradient(135deg, #1e0a0f 0%, #2d1218 100%);
    border: 1px solid rgba(201, 169, 110, 0.35);
    border-radius: 100px;
    padding: 0.55rem 1.4rem;
    margin-top: 1rem;
}
.accuracy-badge .pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #c9a96e;
}
.accuracy-badge .lbl {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #8a7060;
    text-transform: uppercase;
}

/* ── Glass Separator ── */
.glass-sep {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(201,169,110,0.25), transparent);
    margin: 1.5rem 0;
}

/* ── Panel Cards ── */
.panel {
    background: linear-gradient(145deg, rgba(30,10,15,0.8) 0%, rgba(20,8,12,0.9) 100%);
    border: 1px solid rgba(201,169,110,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
.panel-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: #c9a96e;
    letter-spacing: 0.05em;
    margin-bottom: 1.2rem;
}
.panel-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    color: #5a4535;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ── Result Cards ── */
.result-good {
    background: linear-gradient(135deg, #0f2818 0%, #0a1f12 100%);
    border: 1px solid rgba(100, 200, 130, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-bad {
    background: linear-gradient(135deg, #2a0808 0%, #1a0505 100%);
    border: 1px solid rgba(200, 80, 80, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 600;
    margin: 0.3rem 0;
}
.result-verdict-good { color: #6dd68a; }
.result-verdict-bad  { color: #e06060; }
.result-conf {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #8a7060;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.result-conf span {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    color: #c9a96e;
    font-weight: 600;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #100408 0%, #0a0206 100%) !important;
    border-right: 1px solid rgba(201,169,110,0.1) !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: #e8ddd0 !important; }
.sidebar-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    color: #c9a96e;
    font-style: italic;
    margin-bottom: 0.2rem;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(201,169,110,0.08);
    font-size: 0.8rem;
}
.sidebar-stat .sk { color: #5a4535; font-weight: 500; }
.sidebar-stat .sv { color: #c9a96e; font-weight: 600; }

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: #c9a96e !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #8b1a2a 0%, #6b0f1c 100%) !important;
    color: #f5ede0 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(201,169,110,0.3) !important;
    border-radius: 8px !important;
    padding: 0.9rem 2rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #a82035 0%, #8b1a2a 100%) !important;
    border-color: rgba(201,169,110,0.6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(139, 26, 42, 0.4) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(201,169,110,0.04) !important;
    border: 1px solid rgba(201,169,110,0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #5a4535 !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; }
[data-testid="stMetricValue"] { color: #c9a96e !important; font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }

/* ── Section Label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.4em;
    color: #5a4535;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(201,169,110,0.15);
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING - MUST BE IDENTICAL TO TRAIN.PY
# ═══════════════════════════════════════════════════════
def create_features(df_input):
    """Create engineered features - IDENTICAL to train.py"""
    df_new = df_input.copy()

    # Interaction features
    df_new['alcohol_sugar'] = df_new['alcohol'] * df_new['residual sugar']
    df_new['alcohol_acidity'] = df_new['alcohol'] * df_new['fixed acidity']
    df_new['alcohol_volatile'] = df_new['alcohol'] * df_new['volatile acidity']
    df_new['density_alcohol'] = df_new['density'] / (df_new['alcohol'] + 0.001)
    df_new['sulphates_alcohol'] = df_new['sulphates'] * df_new['alcohol']

    # Ratio features
    df_new['free_so2_ratio'] = df_new['free sulfur dioxide'] / (df_new['total sulfur dioxide'] + 1)
    df_new['volatile_fixed_ratio'] = df_new['volatile acidity'] / (df_new['fixed acidity'] + 0.001)
    df_new['citric_fixed_ratio'] = df_new['citric acid'] / (df_new['fixed acidity'] + 0.001)
    df_new['sugar_alcohol_ratio'] = df_new['residual sugar'] / (df_new['alcohol'] + 0.001)
    df_new['chlorides_sulphates_ratio'] = df_new['chlorides'] / (df_new['sulphates'] + 0.001)

    # Polynomial features
    df_new['alcohol_squared'] = df_new['alcohol'] ** 2
    df_new['alcohol_cubed'] = df_new['alcohol'] ** 3
    df_new['sulphates_squared'] = df_new['sulphates'] ** 2
    df_new['volatile_squared'] = df_new['volatile acidity'] ** 2
    df_new['density_squared'] = df_new['density'] ** 2

    # Chemical balance
    df_new['total_acidity'] = df_new['fixed acidity'] + df_new['volatile acidity'] + df_new['citric acid']
    df_new['acidity_to_ph'] = df_new['total_acidity'] * df_new['pH']
    df_new['acidity_balance'] = df_new['fixed acidity'] - df_new['volatile acidity']
    df_new['acid_sugar_balance'] = df_new['total_acidity'] / (df_new['residual sugar'] + 1)

    # Sulfur features
    df_new['bound_so2'] = df_new['total sulfur dioxide'] - df_new['free sulfur dioxide']
    df_new['so2_intensity'] = df_new['total sulfur dioxide'] * df_new['sulphates']

    # Quality flags
    df_new['high_alcohol'] = (df_new['alcohol'] > 11).astype(int)
    df_new['low_volatile'] = (df_new['volatile acidity'] < 0.4).astype(int)
    df_new['optimal_ph'] = ((df_new['pH'] >= 3.0) & (df_new['pH'] <= 3.5)).astype(int)
    df_new['quality_score'] = df_new['high_alcohol'] + df_new['low_volatile'] + df_new['optimal_ph']

    # Log transforms
    df_new['log_residual_sugar'] = np.log1p(df_new['residual sugar'])
    df_new['log_chlorides'] = np.log1p(df_new['chlorides'])
    df_new['log_free_sulfur'] = np.log1p(df_new['free sulfur dioxide'])
    df_new['log_total_sulfur'] = np.log1p(df_new['total sulfur dioxide'])

    return df_new


# ═══════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('model/best_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        metadata = joblib.load('model/model_metadata.pkl')
        bin_mode = joblib.load('model/binary_mode.pkl')
        threshold = joblib.load('model/quality_threshold.pkl')
        return model, scaler, metadata, bin_mode, threshold
    except FileNotFoundError:
        st.error("⚠️ Model not found. Run `python train.py` first.")
        st.stop()

model, scaler, metadata, binary_mode, quality_threshold = load_model_artifacts()


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">Model Information</div>', unsafe_allow_html=True)
    
    stats = [
        ("Algorithm", "Random Forest"),
        ("Accuracy", f"{metadata['test_accuracy']*100:.2f}%"),
        ("Features", str(metadata.get('feature_count', 40))),
        ("Mode", "Binary" if binary_mode else "Multi-class"),
        ("Threshold", f"Quality ≥ {quality_threshold}" if binary_mode else "Scores 3–8"),
    ]
    for k, v in stats:
        st.markdown(f'<div class="sidebar-stat"><span class="sk">{k}</span><span class="sv">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#5a4535;letter-spacing:0.3em;text-transform:uppercase;margin-bottom:0.8rem;">Quality Standards</div>', unsafe_allow_html=True)

    if binary_mode:
        st.markdown(f"""
        <div style='font-size:0.8rem;line-height:2;color:#8a7060;'>
            <span style='color:#6dd68a;'>●</span> <b>Good Wine</b> — Quality ≥ {quality_threshold}<br>
            <span style='color:#e06060;'>●</span> <b>Below Standard</b> — Quality &lt; {quality_threshold}
        </div>""", unsafe_allow_html=True)
    else:
        for score, label, color in [(8, "Exceptional", "#c9a96e"), (7, "Excellent", "#a0d080"),
                                     (6, "Good", "#6dd68a"), (5, "Average", "#d0a060"),
                                     (4, "Below Average", "#d07050"), (3, "Poor", "#e06060")]:
            st.markdown(f'<div style="font-size:0.78rem;color:{color};padding:2px 0;">● {score} — {label}</div>', unsafe_allow_html=True)

    st.markdown('<br><hr style="border-color:rgba(201,169,110,0.1);">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.62rem;color:#3a2520;letter-spacing:0.1em;text-align:center;line-height:1.8;">Powered by Random Forest ML<br>Feature Engineering · SMOTE</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-label">Sommelier Intelligence System</div>
    <div class="hero-title">Vino <em>Intelligence</em></div>
    <div class="hero-line"></div>
    <div class="hero-sub">Machine Learning · Wine Chemistry · Predictive Analysis</div>
    <div style="margin-top:1.2rem;">
        <div class="accuracy-badge">
            <span class="pct">{metadata['test_accuracy']*100:.1f}%</span>
            <div>
                <div class="lbl">Model Accuracy</div>
                <div style="font-size:0.65rem;color:#3a2520;">{metadata.get('model_name','Random Forest')}</div>
            </div>
        </div>
    </div>
</div>
<hr class="glass-sep">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# MAIN PREDICTION SECTION
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-label">Chemical Composition Parameters</div>', unsafe_allow_html=True)

col_l, col_m, col_r = st.columns([1, 1, 1], gap="large")

with col_l:
    st.markdown('<div class="panel-label">Acidity Profile</div>', unsafe_allow_html=True)
    fixed_acidity    = st.slider("Fixed Acidity (g/L)",   3.8, 15.0, 7.5,   0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/L)",0.08, 1.6,  0.5,  0.01)
    citric_acid      = st.slider("Citric Acid (g/L)",     0.0,  1.0,  0.3,  0.01)
    ph               = st.slider("pH",                    2.7,  4.0,  3.3,  0.01)

with col_m:
    st.markdown('<div class="panel-label">Sugar & Minerals</div>', unsafe_allow_html=True)
    residual_sugar = st.slider("Residual Sugar (g/L)",  0.6,  66.0,  2.5,  0.1)
    chlorides      = st.slider("Chlorides (g/L)",       0.009, 0.35,  0.08, 0.001)
    free_sulfur    = st.slider("Free SO₂ (mg/L)",       1.0,  290.0, 15.0, 1.0)
    total_sulfur   = st.slider("Total SO₂ (mg/L)",      6.0,  440.0, 40.0, 1.0)

with col_r:
    st.markdown('<div class="panel-label">Finish & Body</div>', unsafe_allow_html=True)
    density   = st.slider("Density (g/cm³)",  0.987, 1.040, 0.996, 0.0001)
    sulphates = st.slider("Sulphates (g/L)",  0.22,  2.0,   0.65,  0.01)
    alcohol   = st.slider("Alcohol (%ABV)",   8.0,   15.0,  10.5,  0.1)

st.markdown('<br>', unsafe_allow_html=True)

# ── Quick quality indicators ──
alcohol_signal   = "🟢 High" if alcohol > 11 else ("🟡 Medium" if alcohol > 9.5 else "🔴 Low")
volatile_signal  = "🟢 Low"  if volatile_acidity < 0.4 else ("🟡 Medium" if volatile_acidity < 0.6 else "🔴 High")
ph_signal        = "🟢 Optimal" if 3.0 <= ph <= 3.5 else "🟡 Off-range"
so2_ratio        = free_sulfur / (total_sulfur + 1)
so2_signal       = "🟢 Good"  if so2_ratio > 0.25 else "🟡 Check"

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Alcohol Level",    alcohol_signal)
ic2.metric("Volatile Acidity", volatile_signal)
ic3.metric("pH Balance",       ph_signal)
ic4.metric("SO₂ Ratio",        f"{so2_ratio:.2f}")

st.markdown('<br>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("⚗  Analyse Wine Composition")

if analyze:
    raw_input = pd.DataFrame({
        'fixed acidity':        [fixed_acidity],
        'volatile acidity':     [volatile_acidity],
        'citric acid':          [citric_acid],
        'residual sugar':       [residual_sugar],
        'chlorides':            [chlorides],
        'free sulfur dioxide':  [free_sulfur],
        'total sulfur dioxide': [total_sulfur],
        'density':              [density],
        'pH':                   [ph],
        'sulphates':            [sulphates],
        'alcohol':              [alcohol],
    })

    with st.spinner("Running analysis..."):
        # Apply feature engineering
        input_eng = create_features(raw_input)
        input_scaled = scaler.transform(input_eng)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

    st.markdown('<hr class="glass-sep">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)

    res_col, chart_col = st.columns([1, 1], gap="large")

    with res_col:
        is_good = bool(prediction) if binary_mode else prediction >= quality_threshold
        conf    = (proba.max() * 100) if proba is not None else None

        if is_good:
            label = "Good Quality" if binary_mode else f"Score: {prediction}"
            st.markdown(f"""
            <div class="result-good">
                <div class="result-icon">🍾</div>
                <div class="result-verdict result-verdict-good">{label}</div>
                <div style="color:#6dd68a;font-size:0.75rem;letter-spacing:0.15em;margin-top:0.3rem;">
                    MEETS QUALITY STANDARD
                </div>
                {f'<div class="result-conf">Confidence: <span>{conf:.1f}%</span></div>' if conf else ''}
            </div>""", unsafe_allow_html=True)
        else:
            label = "Below Standard" if binary_mode else f"Score: {prediction}"
            st.markdown(f"""
            <div class="result-bad">
                <div class="result-icon">🔬</div>
                <div class="result-verdict result-verdict-bad">{label}</div>
                <div style="color:#e06060;font-size:0.75rem;letter-spacing:0.15em;margin-top:0.3rem;">
                    BELOW QUALITY THRESHOLD
                </div>
                {f'<div class="result-conf">Confidence: <span>{conf:.1f}%</span></div>' if conf else ''}
            </div>""", unsafe_allow_html=True)

    with chart_col:
        if proba is not None:
            if binary_mode:
                labels = [f"Below Standard (<{quality_threshold})", f"Good (≥{quality_threshold})"]
                values = proba
                colors = ['#e06060', '#6dd68a']
            else:
                labels = [str(c) for c in model.classes_]
                values = proba
                colors = px.colors.sequential.Burg_r[:len(labels)]

            fig_donut = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color='#0d0608', width=3)),
                textinfo='label+percent',
                textfont=dict(family='Montserrat', size=11, color='#e8ddd0'),
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                annotations=[dict(
                    text=f"<b>{proba.max()*100:.0f}%</b><br><span style='font-size:9px'>Confidence</span>",
                    font=dict(family='Cormorant Garamond', size=20, color='#c9a96e'),
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # ── Feature influence radar ──
    st.markdown('<hr class="glass-sep"><div class="section-label">Chemical Profile Radar</div>', unsafe_allow_html=True)

    categories = ['Alcohol', 'Acidity', 'Sugar', 'Sulphates', 'pH Balance', 'SO₂ Control']
    raw_values = [
        (alcohol - 8) / 7,
        1 - (volatile_acidity / 1.6),
        1 - min(residual_sugar / 20, 1),
        (sulphates - 0.22) / 1.78,
        1 - abs(ph - 3.25) / 0.75,
        free_sulfur / (total_sulfur + 1),
    ]
    values = [max(0, min(1, v)) for v in raw_values]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(139, 26, 42, 0.25)',
        line=dict(color='#c9a96e', width=2),
        name='Your Wine',
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor='rgba(201,169,110,0.12)', linecolor='rgba(201,169,110,0.12)'),
            angularaxis=dict(gridcolor='rgba(201,169,110,0.1)', linecolor='rgba(201,169,110,0.1)',
                             tickfont=dict(family='Montserrat', size=11, color='#8a7060')),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=300,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ─── FOOTER ─────────────────────────────────────────────────
st.markdown(f"""
<hr class="glass-sep">
<div style="text-align:center;padding:1rem 0 0.5rem;color:#3a2520;font-size:0.65rem;letter-spacing:0.25em;">
    VINO INTELLIGENCE · RANDOM FOREST · {metadata['test_accuracy']*100:.2f}% ACCURACY
    <br><span style="color:#2a1a15;font-size:0.6rem;">SMOTE Balancing · {metadata.get('feature_count', 40)} Engineered Features</span>
</div>
""", unsafe_allow_html=True)
