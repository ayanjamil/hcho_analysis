import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests

st.set_page_config(page_title="HCHO Dashboard", layout="wide")
st.title("📊 OMI-Aura HCHO Dashboard (India)")

# --------------------
# Download + Load India States GeoJSON
# --------------------
@st.cache_resource
def load_states():
    local_path = "india_states.geojson"
    url = "https://raw.githubusercontent.com/datameet/maps/master/geojson/india_states.geojson"

    if not os.path.exists(local_path):
        try:
            st.info("🌍 Downloading India states GeoJSON...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            st.success("✅ GeoJSON downloaded and saved locally!")
        except Exception as e:
            st.error(f"❌ Could not download GeoJSON file: {e}")
            st.stop()

    states = gpd.read_file(local_path)

    # Normalize state column name
    if "ST_NM" in states.columns:
        states.rename(columns={"ST_NM": "State"}, inplace=True)
    elif "NAME_1" in states.columns:
        states.rename(columns={"NAME_1": "State"}, inplace=True)
    elif "st_nm" in states.columns:
        states.rename(columns={"st_nm": "State"}, inplace=True)
    elif "STATE" in states.columns:
        states.rename(columns={"STATE": "State"}, inplace=True)
    else:
        st.error(f"❌ Could not find a state name column. Found: {list(states.columns)}")
        st.stop()

    return states.to_crs(epsg=4326)

india_states = load_states()

# --------------------
# Helper: Plot Charts
# --------------------
def plot_time_series(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["HCHO_Mean (DU)"],
                             mode="lines", name="Mean HCHO (DU)",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["HCHO_Max (DU)"],
                             mode="lines", line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df["date"], y=df["HCHO_Min (DU)"],
                             mode="lines", line=dict(width=0),
                             fill="tonexty",
                             fillcolor="rgba(135,206,250,0.3)",
                             name="Min-Max Range"))
    fig.update_layout(title=title, height=400)
    return fig

def plot_histogram(df, title):
    fig = px.histogram(df, x="HCHO_Mean (DU)", nbins=30,
                       color_discrete_sequence=["green"])
    fig.update_layout(title=title, height=400)
    return fig

def plot_rolling(df, title):
    df = df.copy()
    df["7day_MA"] = df["HCHO_Mean (DU)"].rolling(window=7).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["HCHO_Mean (DU)"],
                             mode="lines", name="Daily Mean",
                             opacity=0.5))
    fig.add_trace(go.Scatter(x=df["date"], y=df["7day_MA"],
                             mode="lines", name="7-day Moving Avg",
                             line=dict(color="red", width=2)))
    fig.update_layout(title=title, height=400)
    return fig

# --------------------
# File Upload
# --------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lat"] = (df["Min_Lat"] + df["Max_Lat"]) / 2
    df["lon"] = (df["Min_Lon"] + df["Max_Lon"]) / 2

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf = gpd.sjoin(gdf, india_states, how="left", predicate="within")

    # --------------------
    # Sidebar Filters
    # --------------------
    st.sidebar.header("🔍 Filters")
    date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])
    map_metric = st.sidebar.selectbox(
        "Select Metric (for Map)",
        [
            "HCHO_Mean (molecules/cm²)",
            "HCHO_Min (molecules/cm²)",
            "HCHO_Max (molecules/cm²)",
            "HCHO_Mean (DU)",
            "HCHO_Min (DU)",
            "HCHO_Max (DU)"
        ]
    )

    mask = (gdf["date"] >= pd.to_datetime(date_range[0])) & (gdf["date"] <= pd.to_datetime(date_range[1]))
    filtered = gdf[mask]

    # Clean DU data
    filtered = filtered[(filtered["HCHO_Mean (DU)"] >= 0) & (filtered["HCHO_Mean (DU)"] <= 5)]
    filtered = filtered.dropna(subset=["HCHO_Mean (DU)", "HCHO_Min (DU)", "HCHO_Max (DU)"])

    # --------------------
    # State-wise HCHO Map
    # --------------------
    state_data = filtered.groupby("State")[map_metric].mean().reset_index()
    map_data = india_states.merge(state_data, on="State", how="left")

    st.subheader("🗺️ State-wise HCHO Distribution")
    fig_map = px.choropleth_mapbox(
        map_data,
        geojson=map_data.geometry,
        locations=map_data.index,
        color=map_metric,
        hover_name="State",
        mapbox_style="carto-positron",
        center={"lat": 22.97, "lon": 78.65},
        zoom=3.8,
        opacity=0.6,
        color_continuous_scale="Viridis"
    )
    fig_map.update_layout(height=800)
    st.plotly_chart(fig_map, use_container_width=True)

    # --------------------
    # India-level Charts (always DU)
    # --------------------
    st.subheader("🇮🇳 India-level Time Series Analysis (Dobson Units)")
    if not filtered.empty:
        fig1 = plot_time_series(filtered, "India – HCHO (DU) with Min–Max Range")
        fig2 = plot_histogram(filtered, "Distribution of Daily Mean HCHO (DU)")
        fig3 = plot_rolling(filtered, "India – HCHO (DU) (7-day Moving Avg)")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ No India-level DU data available for selected filters.")

else:
    st.info("👆 Please upload your dataset to start exploring.")
