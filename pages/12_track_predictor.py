import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_all_data # Assumes load_all_data() handles file loading and caching

# --- Data Loading ---
@st.cache_data
def load_and_analyze_raw_data():
    """Loads the raw song data, calculates percentile popularities, and returns summary stats."""
    try:
        # Load the full dataset (assuming it is available from a CSV file)
        # We need the raw 'Popularity' column to define top/bottom 10%
        df_full = pd.read_csv("top_10000_1950-now.csv")
        
        # Define the thresholds based on the 10th and 90th percentiles of Popularity
        p10 = df_full['Popularity'].quantile(0.10)
        p90 = df_full['Popularity'].quantile(0.90)

        # Filter the groups
        df_top = df_full[df_full['Popularity'] >= p90]
        df_bottom = df_full[df_full['Popularity'] <= p10]
        
        # Define the features to analyze (must match the slider names)
        features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Liveness', 'Speechiness', 'Instrumentalness', 'Loudness', 'Tempo']

        # Calculate average profiles
        avg_top = df_top[features].mean().round(3)
        avg_bottom = df_bottom[features].mean().round(3)
        
        return avg_top, avg_bottom, features

    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot run success predictor.")
        return pd.Series(), pd.Series(), []
    except KeyError:
        st.error("Error: Required columns (Popularity or feature columns) not found in the raw data.")
        return pd.Series(), pd.Series(), []

# Load the comparison data once
avg_top_profile, avg_bottom_profile, features = load_and_analyze_raw_data()


# --- Utility: Normalization for Radar Chart ---
def normalize_feature(feature_name, value):
    """Normalizes specific features (Loudness, Tempo) to a 0-1 scale."""
    if feature_name == 'Loudness':
        # Assuming Loudness ranges roughly from -60 (min) to 0 (max)
        return min(1.0, max(0.0, (value + 60) / 60))
    elif feature_name == 'Tempo':
        # Assuming Tempo ranges roughly from 60 (min) to 200 (max)
        return min(1.0, max(0.0, (value - 60) / 140))
    else:
        # Features already on 0-1 scale
        return value

def create_radar_figure(user_profile, avg_top_profile, avg_bottom_profile, features):
    """Creates a comparative radar chart for the three profiles."""
    fig = go.Figure()
    
    # Names for the traces
    names = ['Your Dream Track', 'Top 10% Hits Avg', 'Bottom 10% Hits Avg']
    profiles = [user_profile, avg_top_profile, avg_bottom_profile]
    
    # Features for the theta axis (radar categories)
    radar_categories = features
    
    for i, profile in enumerate(profiles):
        # Create a list of normalized R values for the radar chart
        r_values = [normalize_feature(f, profile.get(f, 0)) for f in features]
        
        # Create text to show un-normalized values on hover
        hover_text = [f"{f}: {profile.get(f, 0):.3f}" for f in features]

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=radar_categories,
            fill='toself',
            name=names[i],
            hovertemplate='%{theta}: %{text}<extra>%{full_name}</extra>',
            text=hover_text
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Feature Profile Comparison: Your Track vs. Historical Success"
    )
    return fig


# --- Main Page Execution ---
st.title("✨ Track Success Predictor (Conceptual)")
st.header("Compare Your Dream Track's Profile Against Historical Hits")

st.markdown("""
This tool allows you to input custom feature scores for a **'dream song'** and instantly compares its
**feature fingerprint** against the historical average profile of the **Top 10%** and **Bottom 10%**
most popular songs in the entire dataset (1950-Present).
""")

if avg_top_profile.empty:
    st.warning("Cannot run predictor due to missing historical data. Please ensure the raw CSV file is accessible.")
else:
    # --- Feature Input Sliders ---
    st.subheader("1. Define Your Dream Track's Features")
    
    # Organize inputs into two columns for better layout
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    # Left Column: Normalized 0-1 Features
    with col1:
        st.markdown("#### Normalized Features (0.0 to 1.0)")
        user_input['Danceability'] = st.slider('Danceability (Suitability for dancing)', 0.0, 1.0, 0.7, 0.05)
        user_input['Energy'] = st.slider('Energy (Intensity and activity)', 0.0, 1.0, 0.8, 0.05)
        user_input['Valence'] = st.slider('Valence (Musical positivity/mood)', 0.0, 1.0, 0.6, 0.05)
        user_input['Acousticness'] = st.slider('Acousticness (Likelihood of being acoustic)', 0.0, 1.0, 0.1, 0.05)
        user_input['Liveness'] = st.slider('Liveness (Presence of an audience/live recording)', 0.0, 1.0, 0.2, 0.05)
        user_input['Speechiness'] = st.slider('Speechiness (Presence of spoken words/rap)', 0.0, 1.0, 0.1, 0.05)
        user_input['Instrumentalness'] = st.slider('Instrumentalness (Lack of vocals)', 0.0, 1.0, 0.0, 0.05)

    # Right Column: Specific Scale Features
    with col2:
        st.markdown("#### Specific Scale Features")
        user_input['Loudness'] = st.slider('Loudness (dB, typically -60 to 0)', -60.0, 0.0, -5.0, 1.0)
        user_input['Tempo'] = st.slider('Tempo (BPM, typically 60 to 200)', 60, 200, 120, 5)

    # Convert user input to a Pandas Series for consistent comparison
    user_profile_series = pd.Series(user_input)

    st.markdown("---")
    
    # --- Comparison Chart ---
    st.subheader("2. Predicted Success Profile")
    st.info("""
    **Prediction Rule:** The closer your track's profile (blue line) is to the **Top 10% Hits Average** (orange line), 
    the more closely its musical characteristics align with historically successful music.
    """)
    
    fig_comparison = create_radar_figure(user_profile_series, avg_top_profile, avg_bottom_profile, features)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # --- Summary Table ---
    st.subheader("3. Raw Score Comparison Table")
    
    # Combine all three profiles into a single DataFrame for the table display
    df_comparison = pd.DataFrame({
        'Your Dream Track': user_profile_series.round(3),
        'Top 10% Hits Avg': avg_top_profile.round(3),
        'Bottom 10% Hits Avg': avg_bottom_profile.round(3)
    })
    
    st.dataframe(df_comparison, use_container_width=True)
