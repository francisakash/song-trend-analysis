import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import load_all_data

# Load data structures
data = load_all_data()
df_genres = data.get('genres', pd.DataFrame())

st.title("🎵 3. Genre Deep Dive Analysis")

if df_genres.empty:
    st.error("Genre data not loaded. Please ensure 'df_genres' is correctly returned by load_all_data().")
    st.stop()

# --- Compare average features across genres. (Radar Chart) ---
st.subheader("Compare Average Features Across Genres")
st.markdown("The radar chart compares the full 'feature fingerprint' of up to five selected genres, answering the comparison question directly.")

genre_list = df_genres['genre'].unique().tolist()
compare_genres = st.multiselect(
    "Select Genres to Compare:",
    options=genre_list,
    default=['Pop', 'Hip-Hop', 'Classical', 'Jazz'],
    max_selections=5
)

if compare_genres:
    df_compare = df_genres[df_genres['genre'].isin(compare_genres)]
    
    # Select the 7 key audio features for a radar chart
    radar_features = ['danceability', 'energy', 'valence', 'loudness', 'acousticness', 'liveness', 'tempo']
    df_radar = df_compare[['genre'] + radar_features].copy()
    
    # Radar Chart preparation
    fig_radar = go.Figure()
    
    for g in compare_genres:
        data_rows = df_radar[df_radar['genre'] == g]
        if data_rows.empty:
            continue

        data_row = data_rows.iloc[0]
        
        # Manual normalization for loudness and tempo to fit the 0-1 scale
        # Loudness: using a range from -60 (min) to 0 (max)
        norm_loudness = min(1.0, max(0.0, (data_row['loudness'] + 60) / 60))
        # Tempo: using a range from 60 (min) to 200 (approx max)
        norm_tempo = min(1.0, max(0.0, (data_row['tempo'] - 60) / 140))
        
        r_values = [
            data_row['danceability'],
            data_row['energy'],
            data_row['valence'],
            norm_loudness, 
            data_row['acousticness'],
            data_row['liveness'],
            norm_tempo 
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=r_values,
            theta=radar_features,
            fill='toself',
            name=g,
            hovertemplate='%{theta}: %{text}<extra>%{full_name}</extra>',
            text=[f"{data_row[f]:.3f}" for f in radar_features]
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Feature Fingerprint Comparison (Loudness/Tempo Normalized for Scale)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Please select at least one genre to display the feature profile comparison.")

st.markdown("---")
