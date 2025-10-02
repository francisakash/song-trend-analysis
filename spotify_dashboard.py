import streamlit as st
from utils import load_all_data

# --- STREAMLIT DASHBOARD LAYOUT ---

st.set_page_config(
    layout="wide", 
    page_title="Spotify Song Trend Analysis", 
    page_icon="🎧" 
)

st.title("🎧 Spotify Song Trend Analysis Dashboard (1950 - Present)")
st.markdown("### Overview: Key Global Insights")
st.markdown("Navigate using the menu on the left to explore the analysis for specific questions.")

# Note: Load all dataframes to ensure they are cached upon app startup
# This structure is necessary for the KPI calculation below
try:
    data = load_all_data()
    df_temporal = data.get('temporal', st.empty()) # Get temporal data or empty placeholder
    df_genres = data.get('genres', st.empty())     # Get genres data or empty placeholder
except Exception:
    # Handle the case where data might not be fully loaded/available for KPIs
    df_temporal = st.empty()
    df_genres = st.empty()


st.markdown("---")

# --- Key Performance Indicators (KPIs) ---
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Overall Average Danceability
avg_dance = df_genres['danceability'].mean() if not df_genres.empty else "N/A"
col1.metric("Avg. Danceability (Overall)", f"{avg_dance:.3f}" if isinstance(avg_dance, float) else avg_dance, delta_color="off")

# KPI 2: Most Popular Genre
if not df_genres.empty and 'popularity' in df_genres.columns:
    most_popular_genre = df_genres.sort_values('popularity', ascending=False).iloc[0]['genre']
else:
    most_popular_genre = "N/A"
col2.metric("Most Popular Genre", most_popular_genre)

# KPI 3: Total Years Analyzed
if not df_temporal.empty and 'Year' in df_temporal.columns:
    start_year = df_temporal['Year'].min()
    end_year = df_temporal['Year'].max()
    years_analyzed = f"{start_year} - {end_year}"
else:
    years_analyzed = "N/A"
col3.metric("Time Span Analyzed", years_analyzed)

# KPI 4: Loudest Year (for fun)
if not df_temporal.empty and 'Loudness' in df_temporal.columns:
    loudest_year = df_temporal.sort_values('Loudness', ascending=False).iloc[0]
    loudest_info = f"{loudest_year['Loudness']:.2f} dB"
    loudest_year_val = str(loudest_year['Year'])
else:
    loudest_year_val = "N/A"
    loudest_info = "N/A"
col4.metric("Loudest Year", loudest_year_val, loudest_info)

st.markdown("---")

# --- Project Structure & Analysis Pages (10 Pages) ---

st.header("Project Structure & Analysis Pages")
st.markdown("This dashboard contains 10 dedicated pages to explore the evolution of music, grouped into two analysis phases.")

col_page1, col_page2 = st.columns(2)

# Phase 1: Core Trends & Context (Pages 1, 2, 3, 4, 5)
with col_page1:
    st.markdown("#### Phase I: Core Trends & Context")

    st.subheader("1. Temporal Trends")
    st.info("Analyzes how core music features (**Energy, Valence, Loudness**) have evolved on an **annual** basis since 1950, and compares top vs. bottom hits.")
    
    st.subheader("2. Decadal Fingerprint")
    st.info("Identifies the average **decadal profile** (fingerprint) of a popular song, highlighting shifts in listener choice every 10 years.")
    
    st.subheader("3. Genre Deep Dive")
    st.info("Explores how features define different genres using **Radar Charts**, identifying the most energetic or positive genres.")

    st.subheader("4. Feature Correlation")
    st.info("Visualizes the statistical relationship between all audio features using a **Heatmap** to show which traits occur together (e.g., Danceability vs. Energy).")

    st.subheader("5. Top Artist Feature Profile")
    st.info("Analyzes the complete audio feature set (**fingerprint**) for the artists with the most hit songs in the dataset.")
    
# Phase 2: Application, Exploration & Reference (Pages 6, 7, 8, 9, 10)
with col_page2:
    st.markdown("#### Phase II: Application, Exploration & Reference")
    
    st.subheader("6. Popularity Drivers")
    st.info("Investigates the **correlation** between all audio features (like Energy, Danceability) and a track's **popularity score**.")

    st.subheader("7. Statistical Testing")
    st.info("Applies **T-tests** and other methods to definitively answer specific hypotheses (e.g., whether recent songs are significantly more danceable).")

    st.subheader("8. Exploratory Trends")
    st.info("A flexible tool allowing users to **slice and plot trends** by **Year, Genre, or Artist** on demand.")

    st.subheader("9. Track Predictor")
    st.info("A practical application where users can input custom feature scores and benchmark their 'dream song' against the **historical profile of successful music**.")

    st.subheader("10. Data Documentation")
    st.info("Provides a full **glossary of all Spotify audio features**, data source details, and project methodology.")
