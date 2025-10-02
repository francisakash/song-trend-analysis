import streamlit as st
import pandas as pd
from utils import load_all_data # Used to check data integrity

st.title("📚 Data Documentation & Methodology")
st.header("Understanding the Source and Features of the Music Dataset")

# --- 1. Data Source ---
st.subheader("1. Data Source")

st.markdown("""
This dashboard is powered by a compilation of Spotify track data and hit song lists, covering a vast period of popular music history.

* **Primary Dataset:** `top_10000_1950-now.csv` (10,000+ popular songs).
* **Source:** Data is derived from Spotify's Web API, leveraging their analytical features and vast catalog metadata.
* **Scope:** The dataset tracks the evolution of musical characteristics from approximately **1950 to the present day**, focusing on chart-performing or highly streamed tracks.
""")

# --- 2. Methodology & Preparation ---
st.subheader("2. Methodology & Data Preparation")

st.markdown("""
The data was processed into three main structures to facilitate the analysis across the dashboard:

1.  **Raw Song Data:** The base data for single-track queries, correlation analysis, and percentile calculations.
2.  **Temporal Data:** Calculated by taking the **annual mean** of all features, smoothing out year-to-year variation to show macro trends.
3.  **Aggregated Feature Data (Genre/Artist):** Created by taking the **mean average** of all tracks belonging to a specific genre or artist across the entire timeline.
""")


# --- 3. Musical Feature Glossary ---
st.subheader("3. Spotify Audio Feature Glossary")

st.markdown("These are the core metrics (features) provided by Spotify's analysis and used throughout this dashboard:")

col_features, col_scale = st.columns([3, 1])

with col_features:
    st.markdown("""
    | Feature | Description |
    | :--- | :--- |
    | **Danceability** | How suitable a track is for dancing (0.0 = least, 1.0 = most). |
    | **Energy** | Perceptual measure of intensity and activity (0.0 = calm, 1.0 = aggressive/fast). |
    | **Valence** | Musical positivity conveyed by a track (0.0 = sad/negative, 1.0 = happy/positive). |
    | **Acousticness** | A confidence measure of whether the track is acoustic (1.0 = high confidence). |
    | **Liveness** | Detects the presence of an audience in the recording (1.0 = high probability it's live). |
    | **Speechiness** | Detects the presence of spoken words (e.g., talk shows, rap, spoken word). |
    | **Instrumentalness** | Predicts whether a track contains no vocals. |
    | **Loudness** | The overall loudness in decibels (dB), typically ranging from -60 dB (quiet) to 0 dB (loudest). |
    | **Tempo** | The estimated speed of the track in Beats Per Minute (BPM). |
    """)

with col_scale:
    st.markdown("#### Feature Scale")
    st.markdown("""
    | Feature Type | Range |
    | :--- | :--- |
    | **Normalized** | $0.0 - 1.0$ |
    | **Loudness** | $\approx -60 \text{ dB} - 0 \text{ dB}$ |
    | **Tempo** | $\approx 60 \text{ BPM} - 200 \text{ BPM}$ |
    """)
    

# --- 4. Validation Check ---
st.markdown("---")
st.subheader("4. Data Integrity Check")

try:
    # A quick check to show the user the data structure is loaded
    data = load_all_data()
    if data and 'temporal' in data and not data['temporal'].empty:
        st.success(f"Successfully loaded and structured data for temporal analysis. First year: {data['temporal']['Year'].min()}.")
    else:
        st.error("Basic data loading check failed. One or more core DataFrames are empty.")
except Exception as e:
    st.error(f"Error during data integrity check: {e}")
