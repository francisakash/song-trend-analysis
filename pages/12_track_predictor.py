import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # Import numpy for distance calculation
from utils import load_all_data # Assumes load_all_data() handles file loading and caching

# --- Data Loading ---
@st.cache_data
def load_and_analyze_raw_data():
    """
    Loads the raw song data, filters for modern tracks (2010+), calculates percentile popularities,
    and returns summary stats along with dynamic normalization parameters for plotting.
    """
    try:
        # Load the full dataset (assuming it is available from a CSV file)
        df_full = pd.read_csv("top_10000_1950-now.csv")
        
        # --- FIX: Robustly Extract Year from Album Release Date and ensure numeric type ---
        df_full['Year'] = df_full['Album Release Date'].astype(str).str[:4]
        df_full['Year'] = pd.to_numeric(df_full['Year'], errors='coerce')
        df_full.dropna(subset=['Year'], inplace=True) 
        df_full['Year'] = df_full['Year'].astype(int)

        # --- CRITICAL FILTER: Focus only on modern music (2010 onwards) ---
        df_modern = df_full[df_full['Year'] >= 2010]

        # --- FIX: Change thresholds to 5th and 95th percentile for stronger visual separation ---
        # Define the thresholds based on the 5th and 95th percentiles of Popularity
        p05 = df_modern['Popularity'].quantile(0.05)
        p95 = df_modern['Popularity'].quantile(0.95)

        # Filter the groups based on modern data
        df_top = df_modern[df_modern['Popularity'] >= p95]
        df_bottom = df_modern[df_modern['Popularity'] <= p05]
        
        # Define the features to analyze (must match the slider names)
        features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Liveness', 'Speechiness', 'Instrumentalness', 'Loudness', 'Tempo']

        # --- MAJOR FIX: Calculate dynamic min/max for ALL features (0-1 features too) for enhanced separation ---
        normalization_params = {}
        for feature in features:
            normalization_params[feature] = (df_modern[feature].min(), df_modern[feature].max())

        # Calculate average profiles
        avg_top = df_top[features].mean().round(3)
        avg_bottom = df_bottom[features].mean().round(3)
        
        return avg_top, avg_bottom, features, normalization_params # Returning new parameter

    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot run success predictor.")
        return pd.Series(), pd.Series(), [], {}
    except KeyError as e:
        st.error(f"Error: Required column {e} not found in the raw data. Ensure 'Album Release Date' and 'Popularity' exist.")
        return pd.Series(), pd.Series(), [], {}

# Load the comparison data once, capturing the new normalization parameters
avg_top_profile, avg_bottom_profile, features, norm_params = load_and_analyze_raw_data()


# --- Utility: Normalization for Radar Chart ---
def normalize_feature(feature_name, value, norm_params):
    """Normalizes ALL features to a 0-1 scale using dynamic min/max from the modern dataset."""
    
    # Check if the feature has defined min/max parameters
    if feature_name in norm_params:
        min_val, max_val = norm_params[feature_name]
        data_range = max_val - min_val
        
        if data_range > 0:
            # Scale the value from the dataset's observed min to 1.0 (max)
            normalized_value = (value - min_val) / data_range
            # Clip values to ensure they stay between 0.0 and 1.0
            return min(1.0, max(0.0, normalized_value))
        
        # If the range is zero (all values are identical), return a neutral 0.5
        return 0.5 
    
    # Fallback for features not found in normalization_params (should not happen now)
    return value 

def create_radar_figure(user_profile, avg_top_profile, avg_bottom_profile, features, norm_params):
    """Creates a comparative radar chart for the three profiles."""
    fig = go.Figure()
    
    # Names for the traces
    names = ['Your Dream Track', 'Top 5% Super Hits Avg', 'Bottom 5% Flops Avg']
    profiles = [user_profile, avg_top_profile, avg_bottom_profile]
    
    # Features for the theta axis (radar categories)
    radar_categories = features
    
    # Custom colors: Green for user, Red for Top Hits, Gray for Bottom
    colors = ['#1DB954', '#FF4B4B', '#999999'] 
    
    for i, profile in enumerate(profiles):
        # Pass norm_params to normalize_feature
        r_values = [normalize_feature(f, profile.get(f, 0), norm_params) for f in features]
        
        # Create text to show un-normalized values on hover
        hover_text = [f"{f}: {profile.get(f, 0):.3f}" for f in features]

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=radar_categories,
            fill='toself',
            name=names[i],
            line_color=colors[i],
            opacity=0.6 if i != 0 else 0.8, # Make the user track slightly more prominent
            hovertemplate='%{theta}: %{text}<extra>%{full_name}</extra>',
            text=hover_text
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0], # Add visible tick marks
                showline=False # Remove axis line
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
        title_text="Feature Profile Comparison: Your Track vs. Modern Historical Success (2010+)",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# --- New Prediction Function: Calculates Euclidean Distance ---
def calculate_prediction_score(user_profile, avg_top_profile, avg_bottom_profile, features, norm_params):
    """
    Calculates the Euclidean distance between the user's normalized profile
    and the normalized Top 5% and Bottom 5% profiles.
    
    Returns: A score normalized between 0 (closer to flops) and 100 (closer to hits).
    """
    if avg_top_profile.empty or avg_bottom_profile.empty:
        return 0, 0, 0
    
    # 1. Normalize all three profiles using the dynamic min/max ranges
    user_norm = np.array([normalize_feature(f, user_profile.get(f, 0), norm_params) for f in features])
    top_norm = np.array([normalize_feature(f, avg_top_profile.get(f, 0), norm_params) for f in features])
    bottom_norm = np.array([normalize_feature(f, avg_bottom_profile.get(f, 0), norm_params) for f in features])
    
    # 2. Calculate Euclidean Distance
    # Distance from user profile to top hits profile (lower is better)
    dist_to_top = np.linalg.norm(user_norm - top_norm)
    
    # Distance from user profile to bottom hits profile (higher is better)
    dist_to_bottom = np.linalg.norm(user_norm - bottom_norm)
    
    # 3. Calculate Prediction Score (0 to 100)
    # The score should be high if dist_to_bottom is much greater than dist_to_top.
    # We use a simple ratio to determine closeness:
    # Score = dist_to_bottom / (dist_to_top + dist_to_bottom) * 100
    # If dist_to_top is 0 (perfect match to top hits), score = 100
    # If dist_to_bottom is 0 (perfect match to flops), score = 0
    
    # Handle the case where the profiles are identical (shouldn't happen with the current data):
    if dist_to_top + dist_to_bottom == 0:
        prediction_score = 50
    else:
        prediction_score = (dist_to_bottom / (dist_to_top + dist_to_bottom)) * 100
    
    return prediction_score, dist_to_top, dist_to_bottom


# --- Main Page Execution ---
st.title("✨ Track Success Predictor (Conceptual)")
st.header("Benchmark Your Dream Track Against Modern Hits (2010+)")

st.markdown("""
This tool uses historical data from **2010 onwards** to define the most and least successful **feature fingerprints** in the modern era. 
Use the sliders below to create a **'dream song'** and see how closely its characteristics align with tracks that have achieved high popularity.
""")

if avg_top_profile.empty:
    st.warning("Cannot run predictor due to missing historical data. Please ensure the raw CSV file with 'Album Release Date' and 'Popularity' columns is accessible.")
else:
    # --- Feature Input Sliders ---
    st.subheader("1. Define Your Dream Track's Features")
    
    # Organize inputs into two columns for better layout
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    # Left Column: Normalized 0-1 Features
    with col1:
        st.markdown("#### Normalized Features (0.0 to 1.0)")
        user_input['Danceability'] = st.slider('Danceability (Suitability for dancing)', 0.0, 1.0, 0.7, 0.01)
        user_input['Energy'] = st.slider('Energy (Intensity and activity)', 0.0, 1.0, 0.8, 0.01)
        user_input['Valence'] = st.slider('Valence (Musical positivity/mood)', 0.0, 1.0, 0.6, 0.01)
        user_input['Acousticness'] = st.slider('Acousticness (Likelihood of being acoustic)', 0.0, 1.0, 0.1, 0.01)
        user_input['Liveness'] = st.slider('Liveness (Presence of an audience/live recording)', 0.0, 1.0, 0.2, 0.01)
        st.markdown(
            """
            <style>
            [data-testid="stSlider"] {
                margin-bottom: 0px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Right Column: Specific Scale Features
    with col2:
        st.markdown("#### Specific Scale Features")
        user_input['Speechiness'] = st.slider('Speechiness (Presence of spoken words/rap)', 0.0, 1.0, 0.1, 0.01)
        user_input['Instrumentalness'] = st.slider('Instrumentalness (Lack of vocals)', 0.0, 1.0, 0.0, 0.01)
        user_input['Loudness'] = st.slider('Loudness (dB, typically -20 to 0 for modern music)', -20.0, 0.0, -5.0, 0.5)
        user_input['Tempo'] = st.slider('Tempo (BPM, typically 80 to 180)', 80, 180, 120, 1)

    # Convert user input to a Pandas Series for consistent comparison
    user_profile_series = pd.Series(user_input)
    
    # --- Calculate Prediction Score ---
    score, dist_to_top, dist_to_bottom = calculate_prediction_score(
        user_profile_series, avg_top_profile, avg_bottom_profile, features, norm_params
    )
    
    st.markdown("---")
    
    # --- Prediction Output ---
    st.subheader("2. Quantitative Success Prediction")
    
    # Display the score prominently
    score_color = "#1DB954" if score > 50 else "#FF4B4B" if score < 50 else "#FF9900"
    
    st.markdown(
        f"""
        <div style="background-color: #0E1117; padding: 20px; border-radius: 10px; border-left: 5px solid {score_color};">
            <h3 style="color: white; margin-top: 0px;">Predicted Success Score</h3>
            <div style="font-size: 4em; font-weight: bold; color: {score_color}; text-align: center;">
                {score:.1f} / 100
            </div>
            <p style="color: #999999; text-align: center; margin-bottom: 0px;">
            (A score > 50 means your track is closer to the Top 5% Hits profile.)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display the raw distance metrics
    col_dist1, col_dist2 = st.columns(2)
    col_dist1.metric("Distance to Top 5% Hits (Lower is Better)", f"{dist_to_top:.4f}")
    col_dist2.metric("Distance to Bottom 5% Flops (Higher is Better)", f"{dist_to_bottom:.4f}")


    st.markdown("---")
    
    # --- Comparison Chart ---
    st.subheader("3. Feature Profile Visualization")
    st.info("""
    The visualization confirms the closeness of your track (green) to the historical averages. 
    The **Success Score** above provides the precise quantitative prediction based on the distance metric.
    """)
    
    # Pass the new normalization parameters to the chart function
    fig_comparison = create_radar_figure(user_profile_series, avg_top_profile, avg_bottom_profile, features, norm_params)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # --- Summary Table ---
    st.subheader("4. Raw Score Comparison Table")
    
    # Combine all three profiles into a single DataFrame for the table display
    df_comparison = pd.DataFrame({
        'Your Dream Track': user_profile_series.round(3),
        'Top 5% Super Hits Avg (2010+)': avg_top_profile.round(3),
        'Bottom 5% Flops Avg (2010+)': avg_bottom_profile.round(3)
    })
    
    # Update the column names in the display table
    st.dataframe(df_comparison.T.style.format('{:.3f}'), use_container_width=True) # Transpose for better reading

    # Display key averages for context below the graph
    st.markdown("#### Contextual Averages (Modern Hits)")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Avg. Loudness (Top 5%)", f"{avg_top_profile.get('Loudness', 0):.2f} dB")
    col_kpi2.metric("Avg. Danceability (Top 5%)", f"{avg_top_profile.get('Danceability', 0):.3f}")
    col_kpi3.metric("Avg. Energy (Top 5%)", f"{avg_top_profile.get('Energy', 0):.3f}")
