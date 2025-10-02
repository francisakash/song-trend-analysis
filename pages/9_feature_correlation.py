import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import load_all_data

# --- Data Loading (Full Dataset for Detailed Exploration) ---
@st.cache_data
def load_correlation_data():
    """Loads and preprocesses the full dataset for correlation analysis."""
    try:
        # Load the core dataset containing all feature columns
        df = pd.read_csv("top_10000_1950-now.csv")
        
        # Standardize column names for processing
        df.rename(columns={
            'Danceability': 'Danceability', 
            'Energy': 'Energy', 
            'Valence': 'Valence', 
            'Acousticness': 'Acousticness', 
            'Loudness': 'Loudness',
            'Tempo': 'Tempo',
            'Speechiness': 'Speechiness',
            'Liveness': 'Liveness',
            'Instrumentalness': 'Instrumentalness',
            'Popularity': 'Popularity'
        }, inplace=True)
        
        # Define the set of features to analyze
        feature_columns = [
            'Danceability', 'Energy', 'Valence', 'Acousticness', 'Loudness', 
            'Tempo', 'Speechiness', 'Liveness', 'Instrumentalness', 'Popularity'
        ]

        # Convert features to numeric (coercing errors)
        for col in feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[feature_columns].dropna()
    
    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot perform correlation analysis.")
        return pd.DataFrame()

df_corr_data = load_correlation_data()

st.title("🔗 Feature Correlation")
st.header("Understanding the Relationships Between Musical Attributes")
st.markdown("This section explores how different Spotify audio features—from **Energy** and **Valence** to **Loudness** and **Popularity**—relate to one another across the entire dataset.")

if df_corr_data.empty:
    st.stop()

# --- 1. Correlation Heatmap ---
st.subheader("1. Full Feature Correlation Heatmap")
st.markdown("""
The heatmap below shows the **Pearson correlation coefficient** (a value from -1 to 1) between every pair of features.
* **1 (Red):** Perfect positive correlation (Features move together).
* **0 (White):** No linear correlation.
* **-1 (Blue):** Perfect negative correlation (Features move in opposite directions).
""")

# Calculate the correlation matrix
corr_matrix = df_corr_data.corr().round(2)

# Generate Heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlation")
))

fig_heatmap.update_layout(
    title='Correlation Matrix of Key Audio Features and Popularity',
    height=700,
    xaxis_nticks=len(corr_matrix.columns),
    yaxis_nticks=len(corr_matrix.index)
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# --- 2. Scatter Plot Exploration ---
st.subheader("2. Detailed Scatter Plot View")
st.markdown("Select two features below to visualize their relationship using a scatter plot and identify potential clustering or linear trends.")

col_x, col_y = st.columns(2)

feature_options = df_corr_data.columns.tolist()

with col_x:
    x_feature = st.selectbox(
        "Select Feature for X-Axis:", 
        options=feature_options,
        index=feature_options.index('Energy')
    )

with col_y:
    y_feature = st.selectbox(
        "Select Feature for Y-Axis:", 
        options=feature_options,
        index=feature_options.index('Loudness')
    )

if x_feature and y_feature:
    fig_scatter = px.scatter(
        df_corr_data,
        x=x_feature,
        y=y_feature,
        title=f'Relationship between {x_feature} and {y_feature}',
        opacity=0.3,
        height=500
    )
    fig_scatter.update_layout(
        xaxis_title=x_feature.replace('_', ' '),
        yaxis_title=y_feature.replace('_', ' ')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# --- 3. Key Findings (Derived from Heatmap) ---
st.subheader("3. Common Correlation Insights")
st.markdown("""
The heatmap often reveals strong, intuitive relationships in music production:

* **Energy vs. Acousticness:** Usually a strong **negative** correlation. Songs that are highly **Energetic** (fast, loud, active) tend to be less **Acoustic** (unplugged, natural, warm).
* **Energy vs. Loudness:** Typically a strong **positive** correlation. Higher energy levels in a track are almost always achieved with increased **Loudness**.
* **Valence vs. Popularity:** Often a small but notable **positive** correlation, suggesting songs with a more **positive mood (Valence)** tend to be slightly more **Popular**.
""")
