import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Preparedness Index Dashboard", layout="wide")


# ---------- STATS HELPERS ----------
def cohen_d_independent(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled_sd

def interpret_effect_size(d):
    if np.isnan(d):
        return "not defined"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"

# ---------- TITLE ----------
st.markdown(
    """
    <div class="glass-card">
        <h1 style="margin-bottom: 0.25rem;">Household Disaster Preparedness Index</h1>
        <p style="margin-top: 0.2rem; opacity: 0.85;">
            Upload your dataset, map columns, and explore preparedness levels, village differences,
            and statistical tests for your disaster management dissertation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader(
    "Upload data file (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    help="Should include village, 1–3 indicators, and (optionally) an overall score.",
)

if uploaded is None:
    st.info("Upload a file to start the analysis.")
    st.stop()

if uploaded.name.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(uploaded)
else:
    df = pd.read_csv(uploaded)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Data preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

cols = df.columns.tolist()

# ---------- SIDEBAR COLUMN MAPPING ----------
st.sidebar.header("Column mapping")

village_col = st.sidebar.selectbox(
    "Village column",
    cols,
    index=cols.index("Village") if "Village" in cols else 0,
)

indicator_default_candidates = [
    c
    for c in cols
    if any(
        k in c
        for k in [
            "Awareness",
            "Drill_Participation",
            "Shelter_Access",
            "Evacuation_Route_Knowledge",
            "Emergency_Kit",
        ]
    )
]

indicator_cols = st.sidebar.multiselect(
    "Preparedness indicator columns (1–3 scale)",
    cols,
    default=indicator_default_candidates or cols,
)

overall_default = "Overall_Score (Auto Calculate)" if "Overall_Score (Auto Calculate)" in cols else None

overall_col = st.sidebar.selectbox(
    "Overall score column (optional)",
    ["<Compute from indicators>"] + cols,
    index=0,  # default: compute from indicators
)

if len(indicator_cols) == 0:
    st.error("Select at least one indicator column.")
    st.stop()

max_per_indicator = st.sidebar.number_input(
    "Maximum score per indicator",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

threshold_high = st.sidebar.number_input(
    "Threshold for 'high' (high awareness / participation)",
    min_value=1,
    max_value=max_per_indicator,
    value=max_per_indicator,
    step=1,
)

# ---------- DATA PREP ----------
work_df = df.copy()
for c in indicator_cols:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

if overall_col == "<Compute from indicators>":
    work_df["Overall_Score"] = work_df[indicator_cols].sum(axis=1, min_count=1)
    used_overall_col = "Overall_Score"
else:
    work_df[overall_col] = pd.to_numeric(work_df[overall_col], errors="coerce")
    used_overall_col = overall_col

valid = work_df[work_df[used_overall_col].notna()].copy()
n_households = len(valid)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.write(f"**Number of valid households in analysis:** {n_households}")
st.markdown("</div>", unsafe_allow_html=True)

if n_households == 0:
    st.error("No valid overall scores after cleaning. Check column mapping and data.")
    st.stop()

# ---------- CORE METRICS ----------
total_score = valid[used_overall_col].sum()
max_possible_total = n_households * max_per_indicator * len(indicator_cols)
preparedness_index = (total_score / max_possible_total) * 100 if max_possible_total > 0 else np.nan
mean_overall_score = valid[used_overall_col].mean()

if np.isnan(preparedness_index):
    preparedness_category = "Not available"
elif preparedness_index >= 75:
    preparedness_category = "High preparedness"
elif preparedness_index >= 50:
    preparedness_category = "Moderate preparedness"
else:
    preparedness_category = "Low preparedness"

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Mean household preparedness score", f"{mean_overall_score:.2f}")
m2.metric("Preparedness Index", f"{preparedness_index:.1f}%")
m3.metric("Preparedness category", preparedness_category)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="glass-card">
    In thesis-style wording, you could write:
    <br/>
    <i>The Preparedness Index was {preparedness_index:.1f}%, reflecting a relatively {preparedness_category.lower()} level of preparedness in the sample.</i>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- AWARENESS & DRILL PERCENTAGES ----------
awareness_col = next((c for c in indicator_cols if "Awareness" in c), None)
drill_col = next((c for c in indicator_cols if "Drill_Participation" in c), None)

def pct_high(col_name: str):
    if col_name is None or col_name not in valid.columns:
        return np.nan
    series = valid[col_name].dropna()
    if len(series) == 0:
        return np.nan
    return (series >= threshold_high).mean() * 100

pct_high_awareness = pct_high(awareness_col)
pct_drill_participation = pct_high(drill_col)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Awareness and drill participation")
c1, c2 = st.columns(2)
if awareness_col:
    c1.write(f"**% high awareness (score ≥ {threshold_high})**")
    c1.write(f"{pct_high_awareness:.1f}% of households have high awareness.")
else:
    c1.info("No column containing 'Awareness' detected.")

if drill_col:
    c2.write(f"**% high drill participation (score ≥ {threshold_high})**")
    c2.write(f"{pct_drill_participation:.1f}% of households have high drill participation.")
else:
    c2.info("No column containing 'Drill_Participation' detected.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- VILLAGE COMPARISON ----------
village_stats = valid.groupby(village_col)[used_overall_col].agg(
    mean_score="mean", n_households="count"
).reset_index()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Village-level comparison (pivot-style)")

st.dataframe(village_stats)

fig_bar = px.bar(
    village_stats,
    x=village_col,
    y="mean_score",
    color="mean_score",
    color_continuous_scale="Reds",
    title="Average household preparedness score by village",
    labels={village_col: "Village", "mean_score": "Mean preparedness score"},
)
fig_bar.update_layout(
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font_color="#111827",
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- CORRELATION HEATMAP ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Correlation between indicators and overall score")

corr_cols = indicator_cols + [used_overall_col]
corr_df = valid[corr_cols].corr()

z = corr_df.values
x_labels = corr_df.columns.tolist()
y_labels = corr_df.index.tolist()

heatmap = ff.create_annotated_heatmap(
    z,
    x=x_labels,
    y=y_labels,
    colorscale="Reds",
    showscale=True,
    zmin=-1,
    zmax=1,
)
heatmap.update_layout(
    title="Correlation heatmap",
    xaxis=dict(side="bottom"),
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font_color="#111827",
)
st.plotly_chart(heatmap, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- DISTRIBUTION & SCATTER ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Distribution and relationships")

col_a, col_b = st.columns(2)

with col_a:
    fig_hist = px.histogram(
        valid,
        x=used_overall_col,
        nbins=10,
        title="Distribution of overall preparedness scores",
        color_discrete_sequence=["#e11d48"],
    )
    fig_hist.update_layout(
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font_color="#111827",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    x_scatter = st.selectbox(
        "Indicator to compare against overall score",
        indicator_cols,
        key="scatter_indicator",
    )
    fig_scatter = px.scatter(
        valid,
        x=x_scatter,
        y=used_overall_col,
        color=village_col,
        title=f"{x_scatter} vs overall preparedness",
        labels={x_scatter: x_scatter, used_overall_col: "Overall score"},
        color_discrete_sequence=px.colors.sequential.Reds_r,
    )
    fig_scatter.update_layout(
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font_color="#111827",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- T-TEST BETWEEN VILLAGES ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("t-test between two villages (Welch)")

unique_villages = village_stats[village_col].tolist()
if len(unique_villages) < 2:
    st.info("Need at least two villages for a t-test.")
else:
    v1 = st.selectbox("Village 1", unique_villages, index=0)
    v2 = st.selectbox("Village 2", unique_villages, index=1 if len(unique_villages) > 1 else 0)

    data_v1 = valid.loc[valid[village_col] == v1, used_overall_col].dropna()
    data_v2 = valid.loc[valid[village_col] == v2, used_overall_col].dropna()

    if len(data_v1) < 2 or len(data_v2) < 2:
        st.warning("Each village needs at least 2 observations for a reliable t-test.")
    else:
        t_stat, p_val = stats.ttest_ind(data_v1, data_v2, equal_var=False)

        d = cohen_d_independent(data_v1, data_v2)
        d_label = interpret_effect_size(d)

        st.write(f"Mean {v1}: `{data_v1.mean():.2f}` (n = {len(data_v1)})")
        st.write(f"Mean {v2}: `{data_v2.mean():.2f}` (n = {len(data_v2)})")
        st.write(f"t-statistic (Welch): `{t_stat:.3f}`")
        st.write(f"p-value: `{p_val:.4f}`")
        st.write(f"Cohen's d: `{d:.3f}` ({d_label} effect size)")

        if p_val < 0.05:
            st.success("p < 0.05 → Significant difference in preparedness between the two villages.")
        else:
            st.info("p ≥ 0.05 → No statistically significant difference in preparedness between the two villages.")

        st.markdown(
            f"""
            <p style="margin-top: 0.5rem;">
            Example sentence for the dissertation:
            <br/>
            <i>There was {'a significant' if p_val < 0.05 else 'no statistically significant'} difference in household preparedness scores between {v1} and {v2}
            (Welch's t = {t_stat:.2f}, p = {p_val:.3f}, Cohen's d = {d:.2f}, {d_label} effect size).</i>
            </p>
            """,
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

# ---------- SIMPLE CORRELATION SUMMARY ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Indicator–preparedness correlations")

corr_series = corr_df[used_overall_col].drop(labels=[used_overall_col])
corr_table = corr_series.reset_index()
corr_table.columns = ["Indicator", "Correlation with overall score"]

st.dataframe(corr_table)

st.write(
    "You can use these correlations to discuss which preparedness components are most strongly "
    "associated with the overall preparedness score in your dissertation."
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- INFRASTRUCTURE PLACEHOLDER ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Infrastructure condition (placeholder)")
st.write(
    "If your dataset later includes infrastructure variables (e.g., shelter condition, road quality), "
    "you can extend this section with bar charts or stacked bars summarising those indicators by village."
)
st.markdown("</div>", unsafe_allow_html=True)
