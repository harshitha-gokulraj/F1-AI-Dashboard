# ================= IMPORTS =================
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
 
# ================= PAGE CONFIG (must be first) =================
st.set_page_config(
    page_title="PITWALL — F1 Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ================= GLOBAL THEME =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800;900&family=Barlow:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');
 
/* ---- Root Variables ---- */
:root {
    --red: #E8002D;
    --red-dim: #9B001E;
    --white: #F0F0F0;
    --off-white: #B8B8B8;
    --bg: #080808;
    --bg2: #111111;
    --bg3: #181818;
    --bg4: #1F1F1F;
    --border: rgba(255,255,255,0.07);
    --border-bright: rgba(232,0,45,0.35);
    --font-display: 'Barlow Condensed', sans-serif;
    --font-body: 'Barlow', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
}
 
/* ---- App shell ---- */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(232,0,45,0.12) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 79px, rgba(255,255,255,0.02) 80px),
        repeating-linear-gradient(90deg, transparent, transparent 79px, rgba(255,255,255,0.02) 80px);
    font-family: var(--font-body);
    color: var(--white);
}
 
/* ---- Hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
 
/* ---- Top nav bar ---- */
.pitwall-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8,8,8,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    margin-bottom: 0;
}
.pitwall-logo {
    font-family: var(--font-display);
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 0.12em;
    color: var(--white);
    text-transform: uppercase;
}
.pitwall-logo span { color: var(--red); }
.pitwall-pill {
    font-family: var(--font-mono);
    font-size: 10px;
    background: var(--red);
    color: #fff;
    padding: 3px 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.nav-items {
    display: flex;
    gap: 28px;
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: var(--off-white);
    text-transform: uppercase;
}
 
/* ---- Page hero ---- */
.hero-section {
    padding: 3.5rem 2rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: 'F1';
    position: absolute;
    right: -20px;
    top: -40px;
    font-family: var(--font-display);
    font-size: 320px;
    font-weight: 900;
    color: rgba(255,255,255,0.018);
    line-height: 1;
    pointer-events: none;
    letter-spacing: -20px;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--red);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(42px, 6vw, 76px);
    font-weight: 900;
    letter-spacing: -0.01em;
    line-height: 0.95;
    text-transform: uppercase;
    color: var(--white);
    margin: 0;
}
.hero-title em {
    font-style: normal;
    color: var(--red);
}
 
/* ---- Stat cards ---- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.stat-card {
    background: var(--bg2);
    padding: 1.4rem 1.6rem;
    position: relative;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--red), transparent);
}
.stat-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--off-white);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-value {
    font-family: var(--font-display);
    font-size: 40px;
    font-weight: 800;
    color: var(--white);
    line-height: 1;
}
.stat-unit {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--off-white);
    margin-top: 4px;
}
 
/* ---- Section headers ---- */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 1.2rem;
    margin-top: 2.5rem;
}
.section-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--red);
    letter-spacing: 0.25em;
    text-transform: uppercase;
}
.section-title {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--white);
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
 
/* ---- Chart containers ---- */
.chart-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 1px solid var(--border-bright);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.chart-box::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-bright), transparent);
}
 
/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
}
.sidebar-logo {
    font-family: var(--font-display) !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--off-white) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 1rem !important;
    margin-bottom: 1rem !important;
}
 
/* ---- Streamlit components ---- */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [data-testid="stThumbValue"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
}
div[data-testid="metric-container"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-top: 2px solid var(--red);
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--off-white) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    color: var(--white) !important;
}
.stSuccess {
    background: rgba(232,0,45,0.08) !important;
    border: 1px solid rgba(232,0,45,0.35) !important;
    border-left: 3px solid var(--red) !important;
    color: var(--white) !important;
    font-family: var(--font-display) !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--off-white) !important;
}
h1, h2, h3 {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--white) !important;
}
.stDataFrame {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
hr { border-color: var(--border) !important; }
.stMarkdown p { font-family: var(--font-body); color: var(--off-white); }
 
/* ---- Login ---- */
.login-wrap {
    max-width: 400px;
    margin: 10vh auto 0;
    padding: 3rem;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 3px solid var(--red);
}
.login-title {
    font-family: var(--font-display);
    font-size: 36px;
    font-weight: 900;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--white);
    margin-bottom: 4px;
}
.login-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--off-white);
    letter-spacing: 0.2em;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)
 
# ================= PLOTLY THEME =================
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Barlow Condensed, sans-serif', color='#B8B8B8', size=12),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(family='Share Tech Mono', size=11),
        title_font=dict(family='Barlow Condensed', size=13, color='#888')
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(family='Share Tech Mono', size=11),
        title_font=dict(family='Barlow Condensed', size=13, color='#888')
    ),
    legend=dict(
        bgcolor='rgba(17,17,17,0.85)',
        bordercolor='rgba(255,255,255,0.07)',
        borderwidth=1,
        font=dict(family='Barlow Condensed', size=12)
    ),
    margin=dict(l=12, r=12, t=28, b=12),
    colorway=['#E8002D', '#F5A623', '#4A90E2', '#7ED321', '#BD10E0',
              '#9B9B9B', '#D0021B', '#417505', '#4A4A4A', '#B8B8B8']
)
 
F1_COLORS = ['#E8002D', '#FF8700', '#FFD700', '#1E90FF', '#00D2BE',
             '#DC0000', '#FF8000', '#005AFF', '#52E252', '#B6BABD']
 
# ================= LOGIN =================
def login():
    st.markdown("""
    <div style="background:radial-gradient(ellipse 60% 40% at 50% 0%, rgba(232,0,45,0.15) 0%, transparent 70%); min-height:100vh; padding-top:8vh;">
    <div class="login-wrap">
        <div class="login-title">PITWALL</div>
        <div class="login-sub">▸ RESTRICTED ACCESS — AUTHORIZE TO CONTINUE</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.container():
        col_a, col_b, col_c = st.columns([1,2,1])
        with col_b:
            username = st.text_input("USERNAME", placeholder="admin")
            password = st.text_input("PASSWORD", type="password", placeholder="••••••••")
            if st.button("▸  AUTHENTICATE", use_container_width=True):
                if username == "admin" and password == "1234":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("AUTHENTICATION FAILED — CHECK CREDENTIALS")
 
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
 
if not st.session_state['logged_in']:
    login()
    st.stop()
 
# ================= NAV BAR =================
st.markdown("""
<div class="pitwall-nav">
    <div style="display:flex;align-items:center;gap:16px;">
        <div class="pitwall-logo">PIT<span>WALL</span></div>
        <div class="pitwall-pill">LIVE INTEL</div>
    </div>
    <div class="nav-items">
        <span>Analytics</span>
        <span>Drivers</span>
        <span>Circuits</span>
        <span>Predictions</span>
    </div>
    <div style="font-family:var(--font-mono);font-size:10px;color:#555;letter-spacing:0.15em;">F1 INTELLIGENCE PLATFORM</div>
</div>
""", unsafe_allow_html=True)
 
# ================= LOAD DATA =================
@st.cache_data
def load_data():
    results = pd.read_csv("data/results.csv")
    drivers = pd.read_csv("data/drivers.csv")
    races = pd.read_csv("data/races.csv")
    circuits = pd.read_csv("data/circuits.csv")
    df = results.merge(drivers, on="driverId")
    df = df.merge(races, on="raceId")
    df = df.merge(circuits, on="circuitId")
    df['driver_name'] = df['forename'] + " " + df['surname']
    return df
 
df = load_data()
 
# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ PITWALL<br><span style="font-size:10px;color:#555;font-family:var(--font-mono);letter-spacing:0.2em;font-weight:400;">CONTROL DECK</span></div>', unsafe_allow_html=True)
 
    year = st.slider("SEASON", int(df.year.min()), int(df.year.max()), 2020)
    selected_drivers = st.multiselect("DRIVERS", df['driver_name'].unique())
 
    st.markdown("---")
    st.markdown('<div style="font-family:var(--font-mono);font-size:9px;color:#444;letter-spacing:0.2em;">TELEMETRY FEED ACTIVE</div>', unsafe_allow_html=True)
 
filtered_df = df[df['year'] == year]
if selected_drivers:
    filtered_df = filtered_df[filtered_df['driver_name'].isin(selected_drivers)]
 
# ================= HERO =================
st.markdown(f"""
<div class="hero-section">
    <div class="hero-eyebrow">▸ SEASON INTELLIGENCE REPORT</div>
    <h1 class="hero-title">{year}<br><em>FORMULA ONE</em><br>ANALYTICS</h1>
</div>
""", unsafe_allow_html=True)
 
# ================= QUICK STATS =================
cols = st.columns(4)
stats = [
    ("TOTAL RACES", len(df), "RACE ENTRIES"),
    ("DRIVERS", df['driver_name'].nunique(), "UNIQUE COMPETITORS"),
    ("CIRCUITS", df['circuitId'].nunique(), "ACTIVE VENUES"),
    ("SEASONS", df['year'].nunique(), "YEARS OF DATA"),
]
for col, (label, value, unit) in zip(cols, stats):
    with col:
        st.metric(label, f"{value:,}", None)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ================= POSITION DISTRIBUTION =================
st.markdown('<div class="section-head"><span class="section-tag">01 /</span><span class="section-title">Finishing Position Distribution</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
fig1 = px.histogram(
    filtered_df, x="positionOrder", color="driver_name",
    nbins=20, barmode="overlay",
    color_discrete_sequence=F1_COLORS
)
fig1.update_layout(**PLOTLY_THEME, height=340)
fig1.update_traces(opacity=0.85)
st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
 
# ================= PERFORMANCE TREND =================
st.markdown('<div class="section-head"><span class="section-tag">02 /</span><span class="section-title">Race-by-Race Performance Trend</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
trend = filtered_df.groupby(['round', 'driver_name'])['positionOrder'].mean().reset_index()
fig2 = px.line(
    trend, x="round", y="positionOrder",
    color="driver_name", markers=True,
    color_discrete_sequence=F1_COLORS
)
fig2.update_traces(line=dict(width=2), marker=dict(size=6))
fig2.update_yaxes(autorange="reversed", title="POSITION")
fig2.update_xaxes(title="ROUND")
fig2.update_layout(**PLOTLY_THEME, height=340)
st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
 
# ================= CONSISTENCY =================
st.markdown('<div class="section-head"><span class="section-tag">03 /</span><span class="section-title">Consistency Index (Std Dev of Positions)</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
consistency = filtered_df.groupby('driver_name')['positionOrder'].std().reset_index().sort_values('positionOrder')
fig3 = go.Figure(go.Bar(
    x=consistency['driver_name'],
    y=consistency['positionOrder'],
    marker=dict(
        color=consistency['positionOrder'],
        colorscale=[[0, '#E8002D'], [0.5, '#FF8700'], [1, '#1E1E1E']],
        line=dict(width=0)
    )
))
fig3.update_layout(**PLOTLY_THEME, height=320)
fig3.update_xaxes(title="DRIVER")
fig3.update_yaxes(title="STD DEVIATION")
st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("---")
 
# ================= TEAMMATE BATTLE =================
st.markdown('<div class="section-head"><span class="section-tag">04 /</span><span class="section-title">Teammate Head-to-Head</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
team_data = df[df['year'] == year]
team = st.selectbox("SELECT CONSTRUCTOR", team_data['constructorId'].unique())
team_df = team_data[team_data['constructorId'] == team]
drivers_team = team_df['driver_name'].unique()
 
if len(drivers_team) >= 2:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        d1 = st.selectbox("DRIVER 1", drivers_team, key='d1')
    with col_d2:
        d2 = st.selectbox("DRIVER 2", [d for d in drivers_team if d != d1], key='d2')
 
    comp_df = team_df[team_df['driver_name'].isin([d1, d2])]
 
    fig4 = go.Figure()
    colors_d = {'d1': '#E8002D', 'd2': '#4A90E2'}
    for i, driver in enumerate([d1, d2]):
        ddata = comp_df[comp_df['driver_name'] == driver]['positionOrder']
        fig4.add_trace(go.Violin(
            y=ddata, name=driver,
            box_visible=True, meanline_visible=True,
            fillcolor=['rgba(232,0,45,0.18)', 'rgba(74,144,226,0.18)'][i],
            line_color=['#E8002D', '#4A90E2'][i],
            opacity=0.9
        ))
    fig4.update_yaxes(autorange="reversed", title="FINISHING POSITION")
    fig4.update_layout(**PLOTLY_THEME, height=380, violingap=0.3)
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("---")
 
# ================= ML PREDICTIONS =================
st.markdown('<div class="section-head"><span class="section-tag">05 /</span><span class="section-title">AI Prediction Engine</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
ml_col1, ml_col2 = st.columns(2)
 
# --- Finishing Position ---
with ml_col1:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--font-display);font-size:16px;font-weight:700;letter-spacing:0.1em;color:#888;text-transform:uppercase;margin-bottom:1rem;">FINISH POSITION MODEL</div>', unsafe_allow_html=True)
 
    ml_df = df[['grid', 'positionOrder']].dropna()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(ml_df[['grid']], ml_df['positionOrder'])
 
    grid_input = st.slider("GRID POSITION", 1, 20, 5, key='grid1')
    prediction = model.predict([[grid_input]])
    st.success(f"▸  PREDICTED FINISH: P{int(prediction[0])}")
 
    # Feature importance mini chart
    grid_range = list(range(1, 21))
    preds = [int(model.predict([[g]])[0]) for g in grid_range]
    fig_ml1 = go.Figure(go.Scatter(
        x=grid_range, y=preds,
        mode='lines+markers',
        line=dict(color='#E8002D', width=2),
        marker=dict(color='#E8002D', size=5),
        fill='tozeroy',
        fillcolor='rgba(232,0,45,0.07)'
    ))
    fig_ml1.update_layout(**PLOTLY_THEME, height=200,
                           margin=dict(l=0, r=0, t=16, b=0))
    fig_ml1.update_xaxes(title="GRID")
    fig_ml1.update_yaxes(title="PRED POS", autorange="reversed")
    st.plotly_chart(fig_ml1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
# --- Win Probability ---
with ml_col2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--font-display);font-size:16px;font-weight:700;letter-spacing:0.1em;color:#888;text-transform:uppercase;margin-bottom:1rem;">WIN PROBABILITY MODEL</div>', unsafe_allow_html=True)
 
    df['win'] = (df['positionOrder'] == 1).astype(int)
    ml_df2 = df[['grid', 'win']].dropna()
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(ml_df2[['grid']], ml_df2['win'])
 
    grid_input2 = st.slider("GRID POSITION", 1, 20, 3, key='grid2')
    win_prob = model2.predict_proba([[grid_input2]])[0][1]
 
    # Gauge-style progress
    st.success(f"▸  WIN PROBABILITY: {round(win_prob*100, 1)}%")
 
    all_probs = [model2.predict_proba([[g]])[0][1]*100 for g in range(1,21)]
    fig_ml2 = go.Figure(go.Bar(
        x=list(range(1, 21)),
        y=all_probs,
        marker=dict(
            color=all_probs,
            colorscale=[[0,'#1E1E1E'],[0.5,'#FF8700'],[1,'#E8002D']],
            line=dict(width=0)
        )
    ))
    fig_ml2.update_layout(**PLOTLY_THEME, height=200,
                           margin=dict(l=0, r=0, t=16, b=0))
    fig_ml2.update_xaxes(title="GRID")
    fig_ml2.update_yaxes(title="WIN %")
    st.plotly_chart(fig_ml2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("---")
 
# ================= AI DRIVER RANKING =================
st.markdown('<div class="section-head"><span class="section-tag">06 /</span><span class="section-title">AI Composite Driver Ranking</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
rank_df = df.copy()
rank_df['podium'] = (rank_df['positionOrder'] <= 3).astype(int)
rank_df['win'] = (rank_df['positionOrder'] == 1).astype(int)
 
driver_stats = rank_df.groupby('driver_name').agg(
    avg_pos=('positionOrder','mean'),
    avg_grid=('grid','mean'),
    wins=('win','sum'),
    podiums=('podium','sum')
).reset_index()
 
driver_stats['score'] = (
    (1 / driver_stats['avg_pos']) * 0.4 +
    (1 / driver_stats['avg_grid'].replace(0, np.nan).fillna(20)) * 0.2 +
    driver_stats['wins'] * 0.25 +
    driver_stats['podiums'] * 0.15
)
 
driver_stats = driver_stats.sort_values('score', ascending=False).reset_index(drop=True)
top10 = driver_stats.head(10).copy()
top10.index = top10.index + 1
 
rank_col1, rank_col2 = st.columns([3, 2])
 
with rank_col1:
    fig5 = go.Figure()
    colors_rank = ['#E8002D' if i == 0 else '#FF8700' if i == 1 else '#FFD700' if i == 2 else '#444'
                   for i in range(len(top10))]
    fig5.add_trace(go.Bar(
        y=top10['driver_name'][::-1],
        x=top10['score'][::-1],
        orientation='h',
        marker=dict(color=colors_rank[::-1], line=dict(width=0)),
        text=[f"{s:.3f}" for s in top10['score'][::-1]],
        textposition='inside',
        textfont=dict(family='Share Tech Mono', size=11, color='#fff')
    ))
    fig5.update_layout(**PLOTLY_THEME, height=380)
    fig5.update_xaxes(title="COMPOSITE SCORE")
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
with rank_col2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    display_df = top10[['driver_name','wins','podiums','avg_pos','score']].copy()
    display_df.columns = ['DRIVER','WINS','PODIUMS','AVG POS','SCORE']
    display_df['AVG POS'] = display_df['AVG POS'].round(1)
    display_df['SCORE'] = display_df['SCORE'].round(4)
    st.dataframe(display_df, use_container_width=True, height=360)
    st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("---")
 
# ================= CIRCUIT MAP =================
st.markdown('<div class="section-head"><span class="section-tag">07 /</span><span class="section-title">Global Circuit Map</span><span class="section-line"></span></div>', unsafe_allow_html=True)
 
map_df = df[['name','location','country','lat','lng']].drop_duplicates()
 
fig_map = go.Figure(go.Scattergeo(
    lat=map_df['lat'],
    lon=map_df['lng'],
    text=map_df['name'] + '<br>' + map_df['location'] + ', ' + map_df['country'],
    mode='markers',
    marker=dict(
        size=8,
        color='#E8002D',
        symbol='circle',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        opacity=0.9
    ),
    hoverinfo='text'
))
 
fig_map.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    geo=dict(
        showland=True,
        landcolor='#1A1A1A',
        showocean=True,
        oceancolor='#0D0D0D',
        showcoastlines=True,
        coastlinecolor='rgba(255,255,255,0.12)',
        showframe=False,
        bgcolor='rgba(0,0,0,0)',
        projection_type='natural earth',
        showcountries=True,
        countrycolor='rgba(255,255,255,0.06)',
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=440,
    font=dict(family='Share Tech Mono', color='#888')
)
 
st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.plotly_chart(fig_map, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
 
# ================= FOOTER =================
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:3rem;padding:2rem 0;text-align:center;">
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:900;letter-spacing:0.15em;color:rgba(255,255,255,0.15);">PITWALL — F1 INTELLIGENCE PLATFORM</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#333;margin-top:8px;letter-spacing:0.2em;">DATA ACCURACY NOT GUARANTEED FOR RACE DECISIONS</div>
</div>
""", unsafe_allow_html=True)