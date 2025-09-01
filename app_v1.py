import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
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

    # Download if not available locally
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

    # Load GeoJSON into GeoDataFrame
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
# File Upload
# --------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Compute centroid of bounding box
    df["lat"] = (df["Min_Lat"] + df["Max_Lat"]) / 2
    df["lon"] = (df["Min_Lon"] + df["Max_Lon"]) / 2

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )

    # Spatial join: map each point to state
    gdf = gpd.sjoin(gdf, india_states, how="left", predicate="within")

    # --------------------
    # Sidebar Filters
    # --------------------
    st.sidebar.header("🔍 Filters")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["date"].min(), df["date"].max()]
    )

    # Metric filter
    metric = st.sidebar.selectbox(
        "Select Metric",
        [
            "HCHO_Mean (molecules/cm²)",
            "HCHO_Min (molecules/cm²)",
            "HCHO_Max (molecules/cm²)",
            "HCHO_Mean (DU)",
            "HCHO_Min (DU)",
            "HCHO_Max (DU)"
        ]
    )

    # Filter by date
    mask = (gdf["date"] >= pd.to_datetime(date_range[0])) & (gdf["date"] <= pd.to_datetime(date_range[1]))
    filtered = gdf[mask]

    # --------------------
    # Aggregate by State
    # --------------------
    if "State" not in filtered.columns:
        st.error("❌ No state mapping found. Check your GeoJSON file.")
        st.stop()

    state_data = filtered.groupby("State")[metric].mean().reset_index()
    map_data = india_states.merge(state_data, on="State", how="left")

    # --------------------
    # Choropleth Map
    # --------------------
    st.subheader("🗺️ State-wise HCHO Distribution")

    fig = px.choropleth_mapbox(
        map_data,
        geojson=map_data.geometry,
        locations=map_data.index,
        color=metric,
        hover_name="State",
        mapbox_style="carto-positron",
        center={"lat": 22.97, "lon": 78.65},
        zoom=3.8,
        opacity=0.6,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=800)  # increase height (default ~450)
    st.plotly_chart(fig, use_container_width=True)

    # st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Time Series per State
    # --------------------
    st.subheader("📈 Time Series Analysis")

    if not state_data.empty:
        state_selected = st.selectbox("Select a State", state_data["State"].dropna().unique())

        state_ts = filtered[filtered["State"] == state_selected]
        fig2 = px.line(
            state_ts,
            x="date",
            y=metric,
            title=f"{state_selected} - {metric} Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ No data available for selected filters.")

else:
    st.info("👆 Please upload your dataset to start exploring.")
