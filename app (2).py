# ============================================================
#  PLATFORMER WAIT PREDICTIVE MODEL
#  Developed by: Eng. Mohamed Othman
#  Process Design Engineer — api Refinery
#  HOW TO RUN:
#  1. pip install pandas scikit-learn streamlit plotly
#  2. streamlit run codice.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# SETTINGS
# ============================================================
EOC_LIMIT  = 530.0
WARN_LIMIT = 525.0
SOR_DATE   = "2024-03-01"
W1, W2, W3, WT = 6020, 17780, 17360, 41160

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Platformer WAIT Model | Eng. Mohamed Othman",
    page_icon="⚗️",
    layout="wide"
)

st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #070d14 !important; }
section[data-testid="stSidebar"] { background-color: #0e1824 !important; }
/* Text */
.stApp, .stApp p, .stApp label, .stApp div { color: #e8f4ff !important; }
/* Metrics */
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #5a7a99 !important; }
/* Tabs */
.stTabs [data-baseweb="tab"] { color: #5a7a99 !important; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
/* Inputs */
.stNumberInput input, .stTextInput input, .stDateInput input { background: #0a1520 !important; color: #e8f4ff !important; border: 1px solid #1a2d42 !important; }
/* Buttons */
.stDownloadButton button { background: #00d4ff !important; color: #000 !important; font-weight: 700 !important; }
/* Dataframe */
.stDataFrame { background: #0e1824 !important; }
/* Cards/expanders */
[data-testid="stExpander"] { background: #0e1824 !important; border: 1px solid #1a2d42 !important; }
/* Success/Warning/Error */
.stSuccess { background: #0a2a14 !important; border: 1px solid #00e676 !important; }
.stWarning { background: #2a1f00 !important; border: 1px solid #ffb700 !important; }
.stError { background: #2a080d !important; border: 1px solid #ff4455 !important; }
/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #1a2d42; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div style="background:#050e1a;border-left:4px solid #00d4ff;
border-radius:10px;padding:16px 20px;margin-bottom:16px">
<div style="font-family:monospace;font-size:10px;color:#00d4ff;letter-spacing:3px;margin-bottom:4px">
SEMI-REGENERATIVE CATALYTIC REFORMER — api REFINERY — PLATFORMER UNIT ONLY
</div>
<div style="font-size:22px;font-weight:700;color:#e8f4ff">
Platformer WAIT Predictive Model — PR 256 Pt-Re
</div>
<div style="font-family:monospace;font-size:10px;color:#5a7a99;margin-top:2px">
🔒 Locked to Platformer Unit — Not applicable to other units
</div>
<div style="font-family:monospace;font-size:11px;color:#5a7a99;margin-top:4px">
Developed by: <span style="color:#ffb700">Eng. Mohamed Othman</span>
| Process Design Engineer | Gradient Boosting | R²=99.96%
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Settings")
eoc  = st.sidebar.number_input("EOC Limit (°C)",  value=EOC_LIMIT,  step=0.5)
warn = st.sidebar.number_input("Warning (°C)",     value=WARN_LIMIT, step=0.5)
sor  = st.sidebar.text_input("SOR Date",           value=SOR_DATE)
st.sidebar.divider()
st.sidebar.caption("Eng. Mohamed Othman — Process Design Engineer")
st.sidebar.caption("api Refinery | Platformer Unit")

# ============================================================
# FILE UPLOAD
# ============================================================
st.subheader("📂 Upload Data File")
uploaded = st.file_uploader(
    "Upload: Platformer_Data_MohamedOthman.csv",
    type=["csv"],
    help="Required columns: Date, Day, WAIT, Ti1_R2601, To1_R2601, Ti2_R2602, To2_R2602, Ti3_R2603, To3_R2603, Feed_th, Sulfur_ppm, Nitrogen_ppm"
)

if uploaded is None:
    st.info("⬆️ Upload your CSV file to run the model")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(uploaded)
st.success(f"✅ Loaded {len(df)} rows from {uploaded.name}")
sor = st.text_input("SOR Date", value=SOR_DATE)

# ============================================================
# FEATURE ENGINEERING — EXACT SAME LOGIC AS REFERENCE
# ============================================================
df['WAIT_calc']    = (W1*df['Ti1_R2601'] + W2*df['Ti2_R2602'] + W3*df['Ti3_R2603']) / WT
df['Deactivation'] = df['WAIT_calc'] - df['WAIT_calc'].iloc[0]
df['DeltaT1']      = df['Ti1_R2601'] - df['To1_R2601']
df['DeltaT2']      = df['Ti2_R2602'] - df['To2_R2602']
df['DeltaT3']      = df['Ti3_R2603'] - df['To3_R2603']
df['LHSV']         = df['Feed_th'] / 41.16
df['CumSulfur']    = df['Sulfur_ppm'].cumsum()

features = ['Day','Deactivation','DeltaT1','DeltaT2','DeltaT3','LHSV','Sulfur_ppm','Nitrogen_ppm','CumSulfur','H2_HC']

dm = df.dropna(subset=features + ['WAIT'])
X  = dm[features].values
y  = dm['WAIT'].values

# ============================================================
# MODEL — EXACT SAME PARAMETERS AS REFERENCE
# ============================================================
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X, y)
r2 = r2_score(y, model.predict(X))

# ============================================================
# RATE — EXACT SAME OUTLIER LOGIC AS REFERENCE
# ============================================================
last    = dm.iloc[-1]
recent  = dm.tail(100).copy()
mask    = recent['WAIT'].diff().abs() > 2.0
clean   = recent[~mask]
if len(clean) >= 10:
    recent = clean
rate = np.polyfit(recent['Day'].values, recent['WAIT'].values, 1)[0]

# ============================================================
# PREDICTIONS
# ============================================================
last_wait    = float(last['WAIT'])
today         = datetime.today()
last_csv_date = pd.to_datetime(dm['Date'].iloc[-1])
days_from_model = int((eoc  - last_wait) / rate) if rate > 0 else 0
days_warn_model = int((warn - last_wait) / rate) if rate > 0 else 0

# EOC dates ثابتة من آخر CSV
csv_key = f"{uploaded.name}_{len(df)}"
if st.session_state.get('csv_key') != csv_key:
    st.session_state.csv_key   = csv_key
    target_jan15 = datetime(2027, 1, 15)
    days_to_jan  = (target_jan15 - last_csv_date).days
    rate_apc     = (eoc - last_wait) / days_to_jan if days_to_jan > 0 else rate

    # بدون APC
    st.session_state.eoc_date  = (last_csv_date + timedelta(days=int((eoc - last_wait) / rate))).strftime('%Y-%m-%d')
    st.session_state.warn_date = (last_csv_date + timedelta(days=days_warn_model)).strftime('%Y-%m-%d')
    st.session_state.eoc_opt   = (last_csv_date + timedelta(days=int((eoc - last_wait) / (rate*0.7)))).strftime('%Y-%m-%d')
    st.session_state.eoc_pes   = (last_csv_date + timedelta(days=int((eoc - last_wait) / (rate*1.3)))).strftime('%Y-%m-%d')

    # مع APC — Jan 15 ثابت للـ Base و Pessimistic
    # Optimistic أبطأ من rate_apc فبيعدى Jan 15
    st.session_state.eoc_opt_apc  = (last_csv_date + timedelta(days=int((eoc - last_wait) / (rate*0.7)))).strftime('%Y-%m-%d')
    st.session_state.eoc_base_apc = "2027-01-15"
    st.session_state.eoc_pes_apc  = "2027-01-15"

eoc_date     = st.session_state.eoc_date
warn_date    = st.session_state.warn_date
eoc_opt      = st.session_state.eoc_opt
eoc_pes      = st.session_state.eoc_pes
eoc_opt_apc  = st.session_state.eoc_opt_apc
eoc_base_apc = st.session_state.eoc_base_apc
eoc_pes_apc  = st.session_state.eoc_pes_apc
days_to_eoc  = max(0, (pd.to_datetime(eoc_date)  - today).days)
days_to_warn = max(0, (pd.to_datetime(warn_date) - today).days)
days_to_opt  = max(0, (pd.to_datetime(eoc_opt)   - today).days)
days_to_pes  = max(0, (pd.to_datetime(eoc_pes)   - today).days)

# ============================================================
# KPIs
# ============================================================
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("WAIT Now",       f"{last_wait:.1f}°C")
c2.metric("Day (SOR)",      f"{int(last['Day'])}d")
c3.metric("Data Points",    f"{len(dm)}")
c4.metric("Model R²",       f"{r2*100:.2f}%")
c5.metric("Rate/Month",     f"{rate*30:.2f}°C")
c5.caption(f"Rate/Day: {rate:.4f}°C")
c6.metric("Days → Warning", f"{days_to_warn}d  ({warn_date})")
c7.metric("Days → EOC",     f"{days_to_eoc}d  ({eoc_date})")

# ============================================================
# STATUS BANNER
# ============================================================
if days_to_eoc < 60:
    st.error(f"🔴 CRITICAL — EOC in {days_to_eoc} days ({eoc_date}). Plan regeneration immediately!")
elif days_to_eoc < 180:
    st.warning(f"🟡 WARNING — EOC in {days_to_eoc} days ({eoc_date}). Begin planning for regeneration.")
else:
    st.success(f"🟢 STABLE — EOC estimated {eoc_date}. Continue daily monitoring.")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Chart", "🌡 WAIT Calc", "🔮 What-If", "📋 Data", "⚙️ DCS"
])

# ---- TAB 1: CHART ----
with tab1:
    fut_days = list(range(int(last['Day']), int(last['Day'])+400, 3))
    fut_base = [min(last_wait + rate*(d-last['Day']), eoc+15) for d in fut_days]
    fut_opt  = [min(last_wait + rate*0.7*(d-last['Day']), eoc+15) for d in fut_days]
    fut_pes  = [min(last_wait + rate*1.3*(d-last['Day']), eoc+15) for d in fut_days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dm['Day'].tolist(), y=dm['WAIT'].tolist(),
        name='WAIT Actual', line=dict(color='#e8f4ff', width=1.5)))
    fig.add_trace(go.Scatter(x=fut_days, y=fut_base,
        name='Base Forecast', line=dict(color='#00d4ff', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=fut_days, y=fut_opt,
        name='Optimistic', line=dict(color='#00e676', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=fut_days, y=fut_pes,
        name='Pessimistic', line=dict(color='#ff4455', width=1.5, dash='dot')))
    fig.add_hline(y=eoc,  line_dash='dash', line_color='#ff4455', annotation_text=f'EOC {eoc}°C')
    fig.add_hline(y=warn, line_dash='dot',  line_color='#ffb700', annotation_text=f'Warning {warn}°C')
    if days_to_eoc > 0:
        fig.add_vline(x=last['Day']+days_to_eoc, line_dash='dash', line_color='#ff4455')
    fig.update_layout(
        plot_bgcolor='#0e1824', paper_bgcolor='#070d14',
        font=dict(color='#e8f4ff'), height=420,
        yaxis=dict(range=[440, 545]),
        xaxis_title='Day from SOR (2024-03-01)',
        yaxis_title='WAIT (°C)'
    )
    st.plotly_chart(fig, use_container_width=True)

    col_b, col_o, col_p = st.columns(3)
    col_b.metric('Base Case',   f'{days_to_eoc}d',   delta=eoc_date)
    col_o.metric('Optimistic',  f'{days_to_opt}d',   delta=eoc_opt)
    col_p.metric('Pessimistic', f'{days_to_pes}d',   delta=eoc_pes)

    # WAIT_MAX at Jan 15 for each scenario
    st.markdown("---")
    st.subheader("📊 WAIT Max at January 15, 2027 — Three Scenarios")
    target_jan    = datetime(2027, 1, 15)
    days_to_jan   = (target_jan - last_csv_date).days
    w_opt_jan  = round(min(last_wait + rate*0.7 * days_to_jan, eoc), 2)
    w_base_jan = round(min(last_wait + rate     * days_to_jan, eoc), 2)
    w_pes_jan  = round(min(last_wait + rate*1.3 * days_to_jan, eoc), 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Optimistic WAIT_MAX",  f"{w_opt_jan}°C",  delta=f"With APC → {eoc_opt_apc}")
    c2.metric("🔵 Base Case WAIT_MAX",   f"{w_base_jan}°C", delta=f"With APC → {eoc_base_apc} ✅")
    c3.metric("🔴 Pessimistic WAIT_MAX", f"{w_pes_jan}°C",  delta=f"With APC → {eoc_pes_apc} ✅")

    st.caption("WAIT_MAX = maximum WAIT the APC can allow on January 15, 2027 under each scenario to reach EOC on schedule.")

    st.markdown("---")
    st.subheader("📊 Model Fitting — Actual vs Predicted")
    y_pred = model.predict(X)
    errors = dm['WAIT'].values - y_pred

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dm['Day'].tolist(), y=dm['WAIT'].tolist(),
        name='WAIT Actual', line=dict(color='#e8f4ff', width=2)))
    fig2.add_trace(go.Scatter(x=dm['Day'].tolist(), y=y_pred.tolist(),
        name='WAIT Predicted', line=dict(color='#ff6b6b', width=1.5, dash='dot')))
    fig2.update_layout(
        plot_bgcolor='#0e1824', paper_bgcolor='#070d14',
        font=dict(color='#e8f4ff'), height=380,
        title=f'Model Fitting: Actual vs Predicted  |  R²={r2*100:.2f}%  |  Mean Error={abs(errors).mean():.3f}°C  |  Max Error={abs(errors).max():.2f}°C',
        xaxis_title='Day from SOR', yaxis_title='WAIT (°C)',
        yaxis=dict(range=[490, 520]),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig2, use_container_width=True)
    ca, cb, cc = st.columns(3)
    ca.metric("R²", f"{r2*100:.2f}%")
    cb.metric("Mean Error", f"{abs(errors).mean():.3f}°C")
    cc.metric("Max Error", f"{abs(errors).max():.2f}°C")

# ---- TAB 2: WAIT CALC ----
with tab2:
    st.subheader("🌡 Manual WAIT Calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**R-2601**")
        ti1 = st.number_input('T Inlet (°C)',  value=float(last['Ti1_R2601']), step=0.1, key='ti1')
        to1 = st.number_input('T Outlet (°C)', value=float(last['To1_R2601']), step=0.1, key='to1')
    with c2:
        st.markdown("**R-2602**")
        ti2 = st.number_input('T Inlet (°C)',  value=float(last['Ti2_R2602']), step=0.1, key='ti2')
        to2 = st.number_input('T Outlet (°C)', value=float(last['To2_R2602']), step=0.1, key='to2')
    with c3:
        st.markdown("**R-2603**")
        ti3 = st.number_input('T Inlet (°C)',  value=float(last['Ti3_R2603']), step=0.1, key='ti3')
        to3 = st.number_input('T Outlet (°C)', value=float(last['To3_R2603']), step=0.1, key='to3')

    wait_calc = (W1*ti1 + W2*ti2 + W3*ti3) / WT
    contrib1  = W1*ti1/WT
    contrib2  = W2*ti2/WT
    contrib3  = W3*ti3/WT
    dcs_wait  = st.number_input('DCS WAIT Reading (°C)', value=round(last_wait,2))
    diff      = wait_calc - dcs_wait

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Calculated WAIT", f"{wait_calc:.2f}°C")
    col_b.metric("DCS WAIT",        f"{dcs_wait:.2f}°C")
    col_c.metric("Difference",      f"{diff:+.2f}°C")

    if abs(diff) < 2:
        st.success("✅ EXCELLENT MATCH")
    elif abs(diff) < 5:
        st.success("✅ ACCEPTABLE (±5°C)")
    elif abs(diff) < 10:
        st.warning("⚠️ CHECK WEIGHTS")
    else:
        st.error("❌ LARGE DIFFERENCE")

    st.code(f"""
WAIT Formula:
WAIT = (6020 x {ti1:.1f} + 17780 x {ti2:.1f} + 17360 x {ti3:.1f}) / 41160
     = {wait_calc:.2f}°C
Contributions:
  R-2601: {contrib1:.2f}°C  (14.62%)
  R-2602: {contrib2:.2f}°C  (43.20%)
  R-2603: {contrib3:.2f}°C  (42.18%)
""")

# ---- TAB 3: WHAT-IF ----
with tab3:
    st.subheader("🔮 What-If Analysis")
    col1, col2 = st.columns(2)
    with col1:
        curr_wait = st.number_input('Current WAIT (°C)', value=round(last_wait,2))
        curr_rate = rate * 30
        sulfur_chg = st.slider('Sulfur change (%)',   -50, 200, 0)
        nitr_chg   = st.slider('Nitrogen change (%)', -50, 200, 0)
        feed_chg   = st.slider('Feed change (%)',      -20,  20, 0)
    with col2:
        s_factor = 1 + (sulfur_chg/100) * 0.3
        n_factor = 1 + (nitr_chg/100)   * 0.2
        f_factor = 1 + (feed_chg/100)   * 0.1
        new_rate_mo = curr_rate * s_factor * n_factor * f_factor
        new_rate_dy = new_rate_mo / 30
        new_dtE  = int((eoc  - curr_wait) / new_rate_dy) if new_rate_dy > 0 else 0
        base_dtE = int((eoc  - curr_wait) / rate)        if rate > 0        else 0
        delta    = new_dtE - base_dtE
        new_eoc  = (today + timedelta(days=new_dtE)).strftime('%Y-%m-%d')

        st.metric("New Rate/Month",  f"{new_rate_mo:.2f}°C")
        st.metric("New Days to EOC", f"{new_dtE}d  ({new_eoc})", delta=f"{delta:+d} days")
        if delta < 0:
            st.error(f"⚠️ EOC moves {abs(delta)} days EARLIER")
        elif delta > 0:
            st.success(f"✅ EOC moves {delta} days LATER")
        else:
            st.info("No change in EOC date")

# ---- TAB 4: DATA ----
with tab4:
    st.subheader("📋 Data & Feature Importance")
    col_d, col_d2 = st.columns(2)
    with col_d:
        st.metric('Total Rows',  len(dm))
        st.metric('SOR Date',    sor)
        st.metric('WAIT Range',  f"{dm['WAIT'].min():.1f} — {dm['WAIT'].max():.1f}°C")
    with col_d2:
        st.metric('Last Date',   str(dm['Date'].iloc[-1]))
        st.metric('Day Range',   f"{int(dm['Day'].min())} — {int(dm['Day'].max())}")
        st.metric('Rate/Day',    f"{rate:.4f}°C")

    st.dataframe(dm[['Date','Day','WAIT']].tail(20), use_container_width=True)

    imp    = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    imp_df = pd.DataFrame(imp, columns=['Feature','Importance'])
    imp_df['Importance %'] = (imp_df['Importance']*100).round(2)
    st.dataframe(imp_df, use_container_width=True)

# ---- TAB 5: DCS ----
with tab5:
    st.subheader("⚙️ DCS Formula — DeltaV Engineer")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.code(f"""
Block 1 — WAIT from inlet temperatures:
WAIT_CALC = (6020  x TI26305
           + 17780 x TI26312
           + 17360 x TI26316) / 41160

DCS Tags:
  R2601 Inlet: TI26305
  R2602 Inlet: TI26312
  R2603 Inlet: TI26316
""")
    with col_f2:
        st.code(f"""
Block 2 — EOC Prediction:
Rate           = {rate:.4f} °C/day
Days_to_EOC    = ({eoc:.0f} - WAIT_CALC) / Rate
Days_to_Warn   = ({warn:.0f} - WAIT_CALC) / Rate

Block 3 — Status Logic:
IF Days_to_EOC < 60   -> CRITICAL (3)
IF Days_to_EOC < 180  -> WARNING  (2)
ELSE                  -> STABLE   (1)

Current Values:
  WAIT Now     = {last_wait:.2f}°C
  Days to Warn = {days_to_warn} days  ({warn_date})
  Days to EOC  = {days_to_eoc} days  ({eoc_date})
""")

    # EOC Date prominently displayed
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 EOC Date", eoc_date, delta=f"{days_to_eoc} days")
    c2.metric("🟡 Warning Date", warn_date, delta=f"{days_to_warn} days")
    c3.metric("📍 WAIT Now", f"{last_wait:.2f}°C", delta=f"Rate: {rate:.4f}°C/day")
    st.markdown("---")

    st.subheader("🔄 APC Input Equation")
    st.code(f"""
Days_to_EOC = (530 - WAIT_CALC) / {rate:.4f}

where:
WAIT_CALC = (6020 x TI26305 + 17780 x TI26312 + 17360 x TI26316) / 41160
""")

    st.markdown("---")
    st.subheader("🎯 APC Dynamic WAIT_MAX Equation")
    st.code(f"""
// DeltaV Implementation — Two tags:

// Tag 1: WAIT_CALC (already exists in DCS — Block 1)
WAIT_CALC = (6020 x TI26305 + 17780 x TI26312 + 17360 x TI26316) / 41160

// Tag 2: 24-hour rolling average (new tag — eliminates oscillation)
WAIT_24H_AVG = AVERAGE(WAIT_CALC, 1440)

// Daily WAIT_MAX Constraint (auto-updates every day)
WAIT_MAX = WAIT_24H_AVG + ((530 - WAIT_24H_AVG) / (15-Jan-2027 - TODAY))

// APC Constraint Logic
IF WAIT_CALC > WAIT_MAX:
    APC reduces reactor inlet temperature setpoints
ELSE:
    APC free to maximize RON

// Current Values:
//   WAIT_CALC      = {last_wait:.2f}°C
//   Days Remaining = {int((datetime(2027,1,15) - datetime.today()).days)} days
//   WAIT_MAX today = {last_wait + ((530 - last_wait) / int((datetime(2027,1,15) - datetime.today()).days)):.3f}°C
""")

    st.markdown("---")
    st.subheader("📅 WAIT Max Daily Constraint for APC")
    current_wait_input = st.number_input("Current WAIT from DCS (°C)", value=round(last_wait,2), step=0.1)
    target_date    = st.date_input("Target EOC Date", value=datetime(2027, 1, 15).date())
    target_dt      = datetime.combine(target_date, datetime.min.time())
    start_fixed    = datetime(2026, 6, 23)
    days_to_target = (target_dt - start_fixed).days

    if days_to_target > 0:
        rate_req    = (eoc - current_wait_input) / days_to_target
        reduction   = (1 - rate_req / rate) * 100
        fut_dates   = [start_fixed + timedelta(days=i) for i in range(days_to_target+1)]

        # WAIT_MAX بالمعادلة الجديدة الديناميكية
        # WAIT_MAX = WAIT_24H_AVG + ((530 - WAIT_24H_AVG) / أيام_متبقية)
        wait_limits = []
        sim_wait = current_wait_input
        for i in range(days_to_target+1):
            days_remaining = days_to_target - i
            if days_remaining > 0:
                w_max = sim_wait + ((eoc - sim_wait) / days_remaining)
            else:
                w_max = eoc
            wait_limits.append(round(min(w_max, eoc), 3))
            sim_wait = min(sim_wait + rate_req, eoc)

        # 3 سيناريوهات بناءً على Rate الفعلي
        days_base = days_to_target
        days_opt  = days_to_target
        days_pes  = days_to_target
        wait_base = [min(last_wait + rate     * i, eoc) for i in range(days_to_target+1)]
        wait_opt  = [min(last_wait + rate*0.7 * i, eoc) for i in range(days_to_target+1)]
        wait_pes  = [min(last_wait + rate*1.3 * i, eoc) for i in range(days_to_target+1)]

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Required Rate",    f"{rate_req:.4f}°C/day")
        cb.metric("Rate Reduction",   f"{reduction:.1f}%")
        cc.metric("Days to Target",   f"{days_to_target}d")
        cd.metric("Target EOC",       str(target_date))

        fig_apc = go.Figure()
        fig_apc.add_trace(go.Scatter(x=[d.strftime('%Y-%m-%d') for d in fut_dates], y=wait_opt,
            name=f'Optimistic (Rate×0.7) — EOC in {days_opt}d', line=dict(color='#00e676', width=1.5, dash='dot')))
        fig_apc.add_trace(go.Scatter(x=[d.strftime('%Y-%m-%d') for d in fut_dates], y=wait_base,
            name=f'Base Case (Rate×1.0) — EOC in {days_base}d', line=dict(color='#00d4ff', width=2, dash='dash')))
        fig_apc.add_trace(go.Scatter(x=[d.strftime('%Y-%m-%d') for d in fut_dates], y=wait_pes,
            name=f'Pessimistic (Rate×1.3) — EOC in {days_pes}d', line=dict(color='#ff4455', width=1.5, dash='dot')))
        fig_apc.add_trace(go.Scatter(x=[d.strftime('%Y-%m-%d') for d in fut_dates], y=wait_limits,
            name=f'Target Constraint → EOC: {target_date}', line=dict(color='#ffb700', width=2.5)))
        fig_apc.add_hline(y=eoc,  line_dash='dash', line_color='#ff4455', annotation_text=f'EOC {eoc:.0f}°C')
        fig_apc.add_hline(y=warn, line_dash='dot',  line_color='#ffb700', annotation_text=f'Warning {warn:.0f}°C')
        fig_apc.update_layout(
            plot_bgcolor='#0e1824', paper_bgcolor='#070d14',
            font=dict(color='#e8f4ff'), height=420,
            title=f'WAIT Forecast — 3 Scenarios + APC Target Constraint  |  Target EOC: {target_date}',
            xaxis_title='Date', yaxis_title='WAIT (°C)',
            yaxis=dict(range=[last_wait-1, eoc+3])
        )
        st.plotly_chart(fig_apc, use_container_width=True)

        df_apc = pd.DataFrame({
            'Date'          : [d.strftime('%Y-%m-%d') for d in fut_dates],
            'Day_from_Today': list(range(days_to_target+1)),
            'WAIT_Max_C'    : [round(w, 3) for w in wait_limits]
        })
        st.dataframe(df_apc, use_container_width=True)

        csv_apc = df_apc.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Daily WAIT Max CSV", csv_apc, "WAIT_Max_Daily.csv", "text/csv")
    else:
        st.error("Target date must be in the future.")

    st.caption(f"Model: Gradient Boosting | R²={r2*100:.2f}% | Training: {len(dm)} days | SOR: {sor}")
    st.caption("Eng. Mohamed Othman — Process Design Engineer | api Refinery")
