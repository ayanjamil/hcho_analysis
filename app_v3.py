# app_v4.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import requests

st.set_page_config(page_title="HCHO Dashboard (Global)", layout="wide")
st.title("📊 OMI-Aura HCHO Dashboard (Global)")

# --------------------
# Load world countries (cached)
# --------------------
@st.cache_resource
def load_world():
    """Download a Natural Earth GeoJSON (110m) if not present and load as GeoDataFrame."""
    local_path = "ne_110m_admin_0_countries.geojson"
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

    if not os.path.exists(local_path):
        try:
            # Download once and save locally
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            # Bubble up the error so Streamlit stops cleanly
            raise RuntimeError(f"Could not download Natural Earth GeoJSON: {e}")

    world = gpd.read_file(local_path)

    # Normalize country name column
    if "ADMIN" in world.columns:
        world = world.rename(columns={"ADMIN": "country"})
    elif "name" in world.columns:
        world = world.rename(columns={"name": "country"})
    else:
        # fallback: create a country column from available columns
        world["country"] = world.index.astype(str)

    return world.to_crs(epsg=4326)


# call it (will raise if download fails)
try:
    world_countries = load_world()
except Exception as e:
    st.error(f"Failed to load world countries: {e}")
    st.stop()


# --------------------
# Plot helpers
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
                             name="Min–Max Range"))
    fig.update_layout(title=title, height=600, margin=dict(t=50, b=20))
    return fig


def plot_histogram(df, title):
    fig = px.histogram(df, x="HCHO_Mean (DU)", nbins=40,
                       color_discrete_sequence=["green"])
    fig.update_layout(title=title, height=400, margin=dict(t=50, b=20))
    return fig


def plot_rolling(df, title):
    df = df.copy().sort_values("date")
    df["7day_MA"] = df["HCHO_Mean (DU)"].rolling(window=7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["HCHO_Mean (DU)"],
                             mode="lines", name="Daily Mean",
                             opacity=0.5))
    fig.add_trace(go.Scatter(x=df["date"], y=df["7day_MA"],
                             mode="lines", name="7-day Moving Avg",
                             line=dict(color="red", width=2)))
    fig.update_layout(title=title, height=600, margin=dict(t=50, b=20))
    return fig


# --------------------
# File Upload
# --------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if not uploaded_file:
    st.info("👆 Please upload your dataset (CSV) to start exploring.")
    st.stop()

# read CSV
df = pd.read_csv(uploaded_file)
# ensure date column exists
if "date" not in df.columns:
    st.error("CSV must contain a 'date' column.")
    st.stop()

# parse date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Ensure important numeric columns exist & coerce to numeric
expected_cols = [
    "HCHO_Mean (DU)", "HCHO_Min (DU)", "HCHO_Max (DU)",
    "HCHO_Mean (molecules/cm²)", "Min_Lat", "Min_Lon", "Max_Lat", "Max_Lon"
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan
    else:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute center lat/lon robustly (mean of Min/Max if available)
def compute_center_lat_lon(row):
    lat_vals = []
    lon_vals = []
    for c in ("Min_Lat", "Max_Lat"):
        v = row.get(c)
        if pd.notna(v) and -90 <= v <= 90:
            lat_vals.append(v)
    for c in ("Min_Lon", "Max_Lon"):
        v = row.get(c)
        if pd.notna(v) and -180 <= v <= 180:
            lon_vals.append(v)
    if lat_vals and lon_vals:
        return float(np.mean(lat_vals)), float(np.mean(lon_vals))
    return (np.nan, np.nan)

centers = df.apply(compute_center_lat_lon, axis=1)
df["lat"] = [c[0] for c in centers]
df["lon"] = [c[1] for c in centers]

# Build GeoDataFrame of points (only valid coordinates)
gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf_points_valid = gdf_points[~gdf_points.geometry.is_empty & gdf_points.geometry.notnull()].copy()

# Spatial join with countries
if not gdf_points_valid.empty:
    try:
        # geopandas >= 0.10+ supports predicate; older versions use op
        gdf_points_joined = gpd.sjoin(gdf_points_valid, world_countries, how="left", predicate="within")
    except TypeError:
        gdf_points_joined = gpd.sjoin(gdf_points_valid, world_countries, how="left", op="within")
else:
    # empty GeoDataFrame with same index -> ensures later code runs
    gdf_points_joined = gdf_points_valid.copy()
    gdf_points_joined["country"] = np.nan

# --- Safely assign country back into original df by index ---
# gdf_points_joined retains the original index of df rows it matched
if "country" in gdf_points_joined.columns and not gdf_points_joined.empty:
    # Create a series mapping index -> country
    country_series = pd.Series(data=gdf_points_joined["country"].values, index=gdf_points_joined.index)
    # Initialize column if not present
    if "country" not in df.columns:
        df["country"] = np.nan
    # Assign countries to matching indices
    df.loc[country_series.index, "country"] = country_series.values
else:
    if "country" not in df.columns:
        df["country"] = np.nan

# --------------------
# Sidebar Filters
# --------------------
st.sidebar.header("🔍 Filters")
min_date = pd.to_datetime(df["date"].min())
max_date = pd.to_datetime(df["date"].max())
if pd.isna(min_date) or pd.isna(max_date):
    st.sidebar.warning("Dataset missing valid 'date' values.")
    # set fallback range
    min_date = pd.Timestamp("2000-01-01")
    max_date = pd.Timestamp("2100-01-01")

date_range = st.sidebar.date_input("Select Date Range", [min_date.date(), max_date.date()])

aggregation_level = st.sidebar.selectbox("Aggregation level", ["Country (choropleth)", "Point map (scatter)"])
map_metric = st.sidebar.selectbox(
    "Select Metric (for Map)",
    [
        "HCHO_Mean (DU)",
        "HCHO_Min (DU)",
        "HCHO_Max (DU)",
        "HCHO_Mean (molecules/cm²)"
    ]
)

# Filter by date
mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))
filtered = df[mask].copy()

# Clean/validate DU columns
for col in ["HCHO_Mean (DU)", "HCHO_Min (DU)", "HCHO_Max (DU)"]:
    filtered[col] = pd.to_numeric(filtered.get(col), errors="coerce")

# DU sanity filter inputs
du_min = st.sidebar.number_input("DU min (for filtering)", min_value=0.0, value=0.0, step=0.1)
du_max = st.sidebar.number_input("DU max (for filtering)", min_value=0.0, value=5.0, step=0.1)
if "HCHO_Mean (DU)" in filtered.columns:
    # Keep rows where HCHO_Mean (DU) is within bounds OR NaN (so they can still be used for molecule maps)
    filtered = filtered[(filtered["HCHO_Mean (DU)"].between(du_min, du_max)) | (filtered["HCHO_Mean (DU)"].isna())]

# Prepare filtered points
filtered_points = filtered.dropna(subset=["lat", "lon"]).copy()

# --------------------
# Map: Choropleth (country) or Point map
# --------------------
st.subheader("🌍 Global HCHO Map")

if aggregation_level == "Country (choropleth)":
    # Aggregate by country and plot
    country_data = filtered.groupby("country", dropna=True)[map_metric].mean().reset_index()
    world_map = world_countries.merge(country_data, left_on="country", right_on="country", how="left")

    fig_map = px.choropleth_mapbox(
        world_map,
        geojson=world_map.geometry,
        locations=world_map.index,
        color=map_metric,
        hover_name="country",
        mapbox_style="carto-positron",
        center={"lat": 10, "lon": 0},
        zoom=1.2,
        opacity=0.7,
        color_continuous_scale="Viridis",
        labels={map_metric: map_metric}
    )
    fig_map.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("**Top 10 countries by selected metric**")
    ranking = country_data.sort_values(by=map_metric, ascending=False).head(10)
    st.dataframe(ranking.reset_index(drop=True))

else:
    # Point map
    if not filtered_points.empty:
        color_col = map_metric if map_metric in filtered_points.columns else None
        max_points = st.sidebar.slider("Max points on map", 100, 5000, 2000)
        plot_points = filtered_points.head(max_points)

        fig_points = px.scatter_mapbox(
            plot_points,
            lat="lat",
            lon="lon",
            color=color_col,
            hover_name="country",
            hover_data=["date", color_col],
            zoom=1.5,
            height=700,
            opacity=0.7,
            color_continuous_scale="Viridis"
        )
        fig_points.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_points, use_container_width=True)
    else:
        st.warning("⚠️ No valid point coordinates available for the selected date range / filters.")

# --------------------
# Global-level Charts (DU)
# --------------------
st.subheader("📈 Global Time Series & Distributions (Dobson Units)")

du_df = filtered.dropna(subset=["HCHO_Mean (DU)"]).copy()
if not du_df.empty:
    daily = du_df.groupby("date").agg({
        "HCHO_Mean (DU)": "mean",
        "HCHO_Min (DU)": "min",
        "HCHO_Max (DU)": "max"
    }).reset_index()

    fig1 = plot_time_series(daily, "Global – HCHO (DU) with Min–Max Range (daily aggregated)")
    fig2 = plot_histogram(du_df, "Distribution of Daily Mean HCHO (DU) (all samples)")
    fig3 = plot_rolling(daily, "Global – HCHO (DU) (7-day Moving Avg)")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("⚠️ No DU data available for selected filters.")

# --------------------
# Download cleaned / aggregated table
# --------------------
st.markdown("### 🔽 Download filtered dataset")
filtered_for_export = filtered.copy()
if not filtered_for_export.empty:
    filtered_for_export["date"] = pd.to_datetime(filtered_for_export["date"]).dt.strftime("%Y-%m-%d")
    csv = filtered_for_export.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV (filtered)", data=csv, file_name="hcho_filtered_global.csv", mime="text/csv")
else:
    st.info("No data to export for the current filters.")
