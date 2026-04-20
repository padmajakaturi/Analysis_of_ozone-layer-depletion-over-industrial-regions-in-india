# =========================
# PATH SETUP
# =========================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO
from fpdf import FPDF

from preprocessing.preprocess import load_and_clean_data
from models.arima_model import run_arima
from models.lstm_model import run_lstm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Ozone Layer Depletion Dashboard",
    layout="wide"
)

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0F2027, #2C5364);
        padding: 18px;
        border-radius: 10px;
        color: white;
        margin-bottom: 15px;
    ">
    <div style="text-align:center; padding:10px 0;">
        <h1 style="color:white;">🌍 Ozone Layer Depletion Over Industrial Regions in India</h1>
        <h4 style="color:white;">(2022–2025)</h4>
        <p style="font-size:15px; color:white;">
            Machine Learning–based Environmental Monitoring & Risk Analysis Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================
# LOAD DATA
# =========================
df = load_and_clean_data("data/india_ozone_dataset_2022_2025.csv")

# =========================
# REGION SELECTION
regions = ["Select a Region"] + sorted(df["Region"].unique().tolist())

region = st.selectbox("Select Industrial Region", regions)

# Stop execution if user has not selected a real region
if region == "Select a Region":
    st.info("Please select an industrial region to view the analysis.")
    st.stop()

# Only run this when a region is selected
region_df = df[df["Region"] == region]
# =========================
# region = st.sidebar.selectbox("Select Industrial Region", sorted(df["Region"].unique()))
# region_df = df[df["Region"] == region]
# =========================
# REGION HEADING
# =========================
st.markdown(f"## 📍 Selected Region: **{region}**")
st.divider()


# =========================
# RISK LEVEL (DATA DRIVEN)
# =========================
# region_avg = df.groupby("Region")["Ozone_DU"].mean()
# low, high = region_avg.quantile([0.33, 0.66])
# avg_ozone = region_df["Ozone_DU"].mean()

# if avg_ozone <= low:
#     risk_label = "🟥 Danger"
# elif avg_ozone <= high:
#     risk_label = "🟨 Intermediate"
# else:
#     risk_label = "🟩 Safe"
# Percentile thresholds for heatmap (do not remove)
region_avg = df.groupby("Region")["Ozone_DU"].mean()
low, high = region_avg.quantile([0.33, 0.66])
avg_ozone = region_df["Ozone_DU"].mean()
if avg_ozone <= 50:
    risk_label = "🟩 Good"
elif avg_ozone <= 100:
    risk_label = "🟨 Moderate"
elif avg_ozone <= 300:
    risk_label = "🟥 Unhealthy "
elif avg_ozone < 305:
    risk_label = "🟪 Very Unhealthy"
elif avg_ozone >= 305:
    risk_label = "🟫 Hazardous"



# =========================
# METRICS
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Average Ozone (DU)", f"{avg_ozone:.2f}")
c2.metric("Region Risk Level", risk_label)
c3.metric("Years Covered", f"{region_df['Year'].min()}–{region_df['Year'].max()}")
st.divider()

# =========================
# GRAPH 1: INTERACTIVE CFC vs OZONE
# =========================
st.subheader("📊 CFC Emissions vs Ozone Concentration")
fig1 = px.scatter(region_df, x="CFC_ppm", y="Ozone_DU", color="Ozone_DU",
                  size="CFC_ppm", height=300, title="CFC vs Ozone")
st.plotly_chart(fig1, use_container_width=True)
st.caption("Higher CFC emissions correspond to lower ozone concentration.")

# =========================
# GRAPH 2: ENVIRONMENTAL TIME SERIES + TREND
# =========================
st.subheader("📈 Environmental Time Series (2022–2025)")
region_df["Smoothed_Ozone"] = region_df["Ozone_DU"].rolling(2).mean()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=region_df["Year"], y=region_df["Ozone_DU"],
    mode="markers+lines", name="Observed"
))
fig2.add_trace(go.Scatter(
    x=region_df["Year"], y=region_df["Smoothed_Ozone"],
    mode="markers+lines", name="Smoothed Trend"
))
fig2.update_layout(height=300)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Smoothed trend highlights long-term ozone depletion patterns.")

# =========================
# ML MODELS + R² CALCULATION
# =========================
st.subheader("🧠 Model Performance Comparison")
y = region_df["Ozone_DU"].values
X = region_df["Year"].values.reshape(-1, 1)

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)
lr_pred = lr.predict(X)
lr_r2 = r2_score(y, lr_pred)

# ARIMA
arima_rmse, arima_pred = run_arima(y, return_pred=True)
arima_r2 = r2_score(y, arima_pred)

# LSTM
lstm_rmse, lstm_pred = run_lstm(y, return_pred=True)
lstm_r2 = r2_score(y, lstm_pred)

model_df = pd.DataFrame({
    "Model": ["Linear Regression", "ARIMA", "LSTM"],
    "RMSE": [np.sqrt(np.mean((y-lr_pred)**2)), arima_rmse, lstm_rmse],
    "R² Score": [lr_r2, arima_r2, lstm_r2]
})
st.dataframe(model_df, use_container_width=True)

# =========================
# BEST MODEL IDENTIFICATION
# =========================

best_model_row = model_df.loc[model_df["R² Score"].idxmax()]

best_model = best_model_row["Model"]
best_r2 = best_model_row["R² Score"]

accuracy = best_r2 * 100
st.subheader("🏆 Best Performing Model")

st.success(
    f"""
**{best_model}** performed the best for predicting ozone concentration in the selected region.

📊 **Model Accuracy:** {accuracy:.2f}%  
📈 **R² Score:** {best_r2:.3f}

This model provides the most reliable prediction of ozone layer depletion patterns based on the available environmental data.
"""
)
# =========================
# HEATMAP (REGION RISK)
# =========================
st.subheader("🗺️ Ozone Risk Visualization – Selected Region")

left, right = st.columns([2, 1])

# =========================
# HEATMAP (LEFT)
# =========================
with left:
    size = 80
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    mask = radius <= 1

    intensity = np.zeros_like(X)
    variation = np.std(region_df["Ozone_DU"]) * 0.15
    intensity[mask] = avg_ozone * np.exp(-radius[mask] * 2)
    intensity[mask] += np.random.normal(0, variation, intensity[mask].shape)
    intensity[~mask] = np.nan

    fig_heat = px.imshow(
        intensity,
        color_continuous_scale=[
            (0.0, "green"),
            (0.4, "yellow"),
            (0.65, "orange"),
            (1.0, "red")
        ],
        aspect="equal",
        height=360,
        title=f"Ozone Risk Intensity – {region}"
    )

    fig_heat.update_layout(
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        coloraxis_colorbar=dict(title="Ozone Intensity")
    )

    st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# OZONE DEPLETION YEAR ANALYSIS
# =========================
st.subheader("📊 Year-wise Ozone Depletion Analysis")

# Find year with highest depletion (lowest ozone value)
max_depletion_row = region_df.loc[region_df["Ozone_DU"].idxmin()]
max_depletion_year = max_depletion_row["Year"]
max_depletion_value = max_depletion_row["Ozone_DU"]

# Find year with lowest depletion (highest ozone value)
low_depletion_row = region_df.loc[region_df["Ozone_DU"].idxmax()]
low_depletion_year = low_depletion_row["Year"]
low_depletion_value = low_depletion_row["Ozone_DU"]

col1, col2 = st.columns(2)

col1.metric(
    "🔴 Highest Ozone Depletion Year",
    f"{int(max_depletion_year)}",
    f"Ozone: {max_depletion_value:.2f} DU"
)

col2.metric(
    "🟢 Lowest Ozone Depletion Year",
    f"{int(low_depletion_year)}",
    f"Ozone: {low_depletion_value:.2f} DU"
)

st.info(
    "Lower ozone concentration indicates higher ozone layer depletion caused by industrial CFC emissions."
)
# =========================
# TEMPORAL OZONE ANIMATION
# =========================
st.subheader("⏳ Temporal Animation: Ozone Depletion Trend")

# Sort data by year
region_df = region_df.sort_values("Year")

# =========================
# FORECAST NEXT YEAR
# =========================
future_year = np.array([[region_df["Year"].max() + 1]])
forecast_value = lr.predict(future_year)[0]

forecast_df = pd.DataFrame({
    "Year": [future_year[0][0]],
    "Ozone_DU": [forecast_value]
})

# Combine original data + forecast
combined_df = pd.concat([region_df[["Year", "Ozone_DU"]], forecast_df])

# =========================
# YEAR SLIDER
# =========================
selected_year = st.slider(
    "Select Year to Visualize Ozone Level",
    min_value=int(combined_df["Year"].min()),
    max_value=int(combined_df["Year"].max()),
    value=int(combined_df["Year"].min()),
    step=1
)

# Filter data based on slider
year_df = combined_df[combined_df["Year"] <= selected_year]

# =========================
# PLOT GRAPH
# =========================
fig_anim = px.line(
    year_df,
    x="Year",
    y="Ozone_DU",
    markers=True,
    title=f"Ozone Depletion Trend up to {selected_year}"
)

# Highlight forecast point
fig_anim.add_scatter(
    x=[future_year[0][0]],
    y=[forecast_value],
    mode="markers",
    marker=dict(size=12, symbol="star"),
    name="Forecast (Next Year)"
)

st.plotly_chart(fig_anim, use_container_width=True)

st.caption("Use the slider to observe ozone depletion progression year by year including the predicted future trend.")
# =========================
# DATA PANEL (RIGHT)
# =========================
with right:
    # ---- Health Stage ----
    if avg_ozone <= 50:
        stage = "GOOD"
        advisory = "No health impacts expected."
    elif avg_ozone <= 100:
        stage = "MODERATE"
        advisory = "Sensitive individuals should limit prolonged outdoor activity."
    elif avg_ozone <= 150:
        stage = "UNHEALTHY (Sensitive Groups)"
        advisory = "Children, elderly, and asthma patients should reduce exertion."
    elif avg_ozone <= 200:
        stage = "UNHEALTHY"
        advisory = "Everyone should limit outdoor exertion."
    elif avg_ozone <= 300:
        stage = "VERY UNHEALTHY"
        advisory = "All groups should avoid outdoor activity."
    else:
        stage = "HAZARDOUS"
        advisory = "Health emergency conditions."

    # ---- Trend ----
    trend = np.polyfit(region_df["Year"], region_df["Ozone_DU"], 1)[0]
    if trend > 0:
        arrow = "⬆ Increasing"
    elif trend < 0:
        arrow = "⬇ Decreasing"
    else:
        arrow = "➡ Stable"

    st.markdown(f"### 📍 {region}")
    st.metric("Average Ozone (DU)", f"{avg_ozone:.2f}")
    st.markdown(f"**Stage:** {stage}")
    st.markdown(f"**Trend:** {arrow}")
    st.markdown(f"**Health Advisory:** {advisory}")

st.caption(
    "Left: Circular heatmap showing intra-regional ozone intensity. "
    "Right: Region-wise ozone concentration, health stage, and trend."
)

# =========================
# WARNINGS
# =========================
st.subheader("⚠ Environmental & Health Advisory")

if stage == "GOOD":
    st.success("Air quality is satisfactory. No health impacts expected.")

elif stage == "MODERATE":
    st.warning("Unusually sensitive people should limit prolonged outdoor activity.")

elif "Sensitive" in stage:
    st.warning("Sensitive groups should reduce prolonged or heavy outdoor activity.")

elif stage == "UNHEALTHY":
    st.error("Everyone should limit outdoor exertion. Sensitive groups should avoid it.")

elif stage == "VERY UNHEALTHY":
    st.error("Health alert! All groups should avoid all outdoor physical activity.")

elif stage == "HAZARDOUS":
    st.error("🚨 Health emergency! Everyone is likely to be affected.")

st.divider()



# =========================
# SUGGESTIONS (REGION-SPECIFIC)
# =========================
st.subheader("✅ Suggestions & Recommendations")

if "Good" in risk_label:
    st.success(f"""
**Air Quality Status: Good (🟢)**  

✔ Maintain current industrial emission standards  
✔ Continue periodic ozone monitoring  
✔ Promote green certifications for industries  
✔ Encourage public awareness on ozone protection  
""")

elif "Moderate" in risk_label:
    st.warning(f"""
**Air Quality Status: Moderate (🟡)**  

⚠ Unusually sensitive individuals should limit prolonged outdoor activity  
✔ Strengthen emission audits in industrial zones  
✔ Increase frequency of ozone monitoring  
✔ Encourage industries to adopt low-CFC alternatives  
""")

elif "Unhealthy" in risk_label:
    st.error(f"""
**Air Quality Status: Unhealthy (🔴)**  

🚨 Everyone should limit prolonged outdoor activity  
✔ Immediate reduction of industrial CFC emissions  
✔ Temporary restrictions on high-polluting units  
✔ Deploy emergency air-quality response plans  
""")

elif "Very Unhealthy" in risk_label:
    st.error(f"""
**Air Quality Status: Very Unhealthy (🟣)**  

🚨 Health alert for all population groups  
✔ Suspend non-essential industrial operations  
✔ Issue public advisories and school restrictions  
✔ Implement emergency ozone mitigation strategies  
""")

elif "Hazardous" in risk_label:
    st.error(f"""
**Air Quality Status: Hazardous (🟤)**  

🚨 Severe health emergency – everyone is at risk  
✔ Immediate shutdown of major polluting industries  
✔ Activate disaster-level environmental response  
✔ Continuous real-time monitoring and evacuation advisories  
""")

st.divider()

