import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
from utils import load_all_data # Assumes load_all_data() handles file loading and caching

# --- Configuration ---
SUCCESS_THRESHOLD = 80 # New definition: Popularity must be 80 or higher
FEATURES = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Liveness', 'Speechiness', 
            'Instrumentalness', 'Loudness', 'Tempo']

# --- Data Loading ---
@st.cache_data
def load_and_analyze_raw_data():
    """
    Loads the raw song data, filters for modern tracks (2010+), defines the 
    'Success Profile' (Popularity >= 80), and returns the profile and normalization parameters.
    """
    try:
        # Load the full dataset (assuming it is available from a CSV file)
        df_full = pd.read_csv("top_10000_1950-now.csv")
        
        # Robustly extract Year and filter for modern music
        df_full['Year'] = df_full['Album Release Date'].astype(str).str[:4]
        df_full['Year'] = pd.to_numeric(df_full['Year'], errors='coerce')
        df_full.dropna(subset=['Year'], inplace=True) 
        df_full['Year'] = df_full['Year'].astype(int)
        df_modern = df_full[df_full['Year'] >= 2010]

        if df_modern.empty:
             st.error("No modern track data found (2010+).")
             return pd.Series(), FEATURES, {}
        
        # --- DEFINE SUCCESS PROFILE: Popularity >= 80 ---
        df_hits = df_modern[df_modern['Popularity'] >= SUCCESS_THRESHOLD]
        
        if df_hits.empty:
            st.error(f"No tracks found with Popularity >= {SUCCESS_THRESHOLD}.")
            return pd.Series(), FEATURES, {}

        # Calculate dynamic min/max for ALL features across the ENTIRE modern dataset
        normalization_params = {
            feature: (df_modern[feature].min(), df_modern[feature].max()) 
            for feature in FEATURES
        }

        # Calculate the average profile for the hits
        avg_hit_profile = df_hits[FEATURES].mean().round(3)
        
        return avg_hit_profile, FEATURES, normalization_params

    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot run success predictor.")
        return pd.Series(), FEATURES, {}
    except KeyError as e:
        st.error(f"Error: Required column {e} not found in the raw data.")
        return pd.Series(), FEATURES, {}

# Load the comparison data
avg_hit_profile, features, norm_params = load_and_analyze_raw_data()


# --- Utility: Normalization and Charting ---
def normalize_feature(feature_name, value, norm_params):
    """Normalizes ALL features to a 0-1 scale using dynamic min/max from the modern dataset."""
    if feature_name in norm_params:
        min_val, max_val = norm_params[feature_name]
        data_range = max_val - min_val
        
        if data_range > 0:
            normalized_value = (value - min_val) / data_range
            return min(1.0, max(0.0, normalized_value))
        return 0.5 
    return value 

def create_radar_figure(user_profile, avg_hit_profile, features, norm_params):
    """Creates a comparative radar chart for the user profile against the hit profile."""
    fig = go.Figure()
    
    names = ['Your Dream Track', f'Avg Popularity >= {SUCCESS_THRESHOLD} Hit Profile']
    profiles = [user_profile, avg_hit_profile]
    
    # Custom colors: Green for user, Red for Hit Profile
    colors = ['#1DB954', '#FF4B4B'] 
    
    for i, profile in enumerate(profiles):
        r_values = [normalize_feature(f, profile.get(f, 0), norm_params) for f in features]
        hover_text = [f"{f}: {profile.get(f, 0):.3f}" for f in features]

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=features,
            fill='toself',
            name=names[i],
            line_color=colors[i],
            opacity=0.6 if i != 0 else 0.8,
            hovertemplate='%{theta}: %{text}<extra>%{full_name}</extra>',
            text=hover_text
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                showline=False
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        title_text=f"Feature Profile Comparison: Your Track vs. Modern Hits (Popularity $\geq {SUCCESS_THRESHOLD}$)",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# --- Prediction Function: Calculates Similarity Score ---
def calculate_prediction_score(user_profile, avg_hit_profile, features, norm_params):
    """
    Calculates the Euclidean distance between the user's normalized profile
    and the normalized Hit Profile.
    
    Returns: A score normalized between 0 (low similarity) and 100 (perfect similarity).
    """
    if avg_hit_profile.empty:
        return 0, 0
    
    # 1. Normalize both profiles
    user_norm = np.array([normalize_feature(f, user_profile.get(f, 0), norm_params) for f in features])
    hit_norm = np.array([normalize_feature(f, avg_hit_profile.get(f, 0), norm_params) for f in features])
    
    # 2. Calculate Euclidean Distance (lower is better)
    dist_to_hit = np.linalg.norm(user_norm - hit_norm)
    
    # 3. Calculate Similarity Score (0 to 100)
    # The maximum possible distance is fixed (the distance between all 0s and all 1s on a 9-feature radar chart)
    MAX_POSSIBLE_DISTANCE = np.sqrt(len(features)) # Max distance in an N-dimensional unit cube
    
    # Normalize distance to 0-1 scale (0 is max distance, 1 is 0 distance)
    # Score = 1 - (Actual Distance / Max Possible Distance) * 100
    
    similarity_ratio = 1 - (dist_to_hit / MAX_POSSIBLE_DISTANCE)
    prediction_score = similarity_ratio * 100
    
    # Ensure score is clipped between 0 and 100
    return min(100.0, max(0.0, prediction_score)), dist_to_hit

# --- Main Page Execution ---
st.title("✨ Targeted Track Hit Predictor")
st.header(f"Similarity Benchmark Against Super Hits (Popularity $\geq {SUCCESS_THRESHOLD}$)")

st.markdown(f"""
This tool determines your track's potential for being a massive hit by measuring the **similarity** of its audio features to the average characteristics of all modern (2010+) songs with a **Popularity score of {SUCCESS_THRESHOLD} or higher**.

**Prediction Rule:** The higher the **Similarity Score (0-100)**, the more closely your track aligns with historically successful hit music.
""")

if avg_hit_profile.empty:
    st.warning("Cannot run predictor due to missing historical data.")
else:
    # --- Feature Input Sliders ---
    st.subheader("1. Define Your Dream Track's Features")
    
    col1, col2 = st.columns(2)
    user_input = {}
    
    # Left Column: Normalized 0-1 Features
    with col1:
        st.markdown("#### Normalized Features (0.0 to 1.0)")
        user_input['Danceability'] = st.slider('Danceability', 0.0, 1.0, 0.7, 0.01)
        user_input['Energy'] = st.slider('Energy', 0.0, 1.0, 0.8, 0.01)
        user_input['Valence'] = st.slider('Valence', 0.0, 1.0, 0.6, 0.01)
        user_input['Acousticness'] = st.slider('Acousticness', 0.0, 1.0, 0.1, 0.01)
        user_input['Liveness'] = st.slider('Liveness', 0.0, 1.0, 0.2, 0.01)

    # Right Column: Specific Scale Features
    with col2:
        st.markdown("#### Specific Scale Features")
        user_input['Speechiness'] = st.slider('Speechiness', 0.0, 1.0, 0.1, 0.01)
        user_input['Instrumentalness'] = st.slider('Instrumentalness', 0.0, 1.0, 0.0, 0.01)
        user_input['Loudness'] = st.slider('Loudness (dB)', -20.0, 0.0, -5.0, 0.5)
        user_input['Tempo'] = st.slider('Tempo (BPM)', 80, 180, 120, 1)

    # Convert user input to a Pandas Series
    user_profile_series = pd.Series(user_input)
    
    # --- Calculate Prediction Score ---
    score, dist_to_hit = calculate_prediction_score(
        user_profile_series, avg_hit_profile, features, norm_params
    )
    
    st.markdown("---")
    
    # --- Prediction Output ---
    st.subheader("2. Quantitative Similarity Prediction")
    
    # Display the score prominently
    score_color = "#1DB954" if score >= 65 else "#FF9900" if score >= 40 else "#FF4B4B"
    score_text = "HIT POTENTIAL" if score >= 65 else "MODERATE ALIGNMENT" if score >= 40 else "LOW ALIGNMENT"
    
    st.markdown(
        f"""
        <div style="background-color: #0E1117; padding: 20px; border-radius: 10px; border-left: 5px solid {score_color};">
            <h3 style="color: white; margin-top: 0px;">Predicted Similarity Score</h3>
            <div style="font-size: 4em; font-weight: bold; color: {score_color}; text-align: center;">
                {score:.1f} / 100
            </div>
            <p style="color: #999999; text-align: center; margin-bottom: 5px;">
            **Prediction:** {score_text}
            </p>
            <p style="color: #999999; text-align: center; margin-bottom: 0px;">
            (Score measures closeness to the average profile of tracks with Popularity $\geq {SUCCESS_THRESHOLD}$)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the raw distance metric
    st.metric("Euclidean Distance to Hit Profile (Lower is Better)", f"{dist_to_hit:.4f}")


    st.markdown("---")
    
    # --- Comparison Chart ---
    st.subheader("3. Feature Profile Visualization")
    st.info(f"""
    The visualization shows how your track's shape (green) compares to the **Avg Hit Profile (red)**.
    The goal is to overlap the red line as closely as possible.
    """)
    
    fig_comparison = create_radar_figure(user_profile_series, avg_hit_profile, features, norm_params)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # --- Summary Table ---
    st.subheader("4. Raw Feature Comparison Table")
    
    df_comparison = pd.DataFrame({
        'Your Dream Track': user_profile_series.round(3),
        f'Avg Popularity $\geq {SUCCESS_THRESHOLD}$ Hit Profile': avg_hit_profile.round(3)
    })
    
    st.dataframe(df_comparison.T.style.format('{:.3f}'), use_container_width=True)
    
