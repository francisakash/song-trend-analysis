import streamlit as st
import pandas as pd
import plotly.express as px

# --- Data Loading and Caching ---

@st.cache_data
def load_temporal_data():
    """Loads and returns the yearly trend data."""
    df = pd.read_csv("temporal_trends.csv")
    return df

@st.cache_data
def load_genre_data():
    """Loads and returns the genre profile data."""
    df = pd.read_csv("genre_profiles.csv")
    return df

@st.cache_data
def load_decadal_data():
    """Loads and returns the decade-grouped feature mean data for Q5."""
    df = pd.read_csv("decadal_trends.csv")
    df['Decade_Label'] = df['Decade'].astype(str) + 's'
    return df

@st.cache_data
def load_top_artists_features():
    """Loads and returns the top 30 artists' average features (all audio features)."""
    df = pd.read_csv("top_30_artists_features.csv") 
    return df

@st.cache_data
def load_popularity_correlations():
    """Loads the correlation coefficients between features and popularity."""
    df = pd.read_csv("popularity_correlations.csv")
    return df

@st.cache_data
def load_song_clusters():
    """Loads the cluster analysis results (k-means output)."""
    df = pd.read_csv("song_clusters.csv") 
    return df

@st.cache_data
def load_statistical_results():
    """Loads the results from the t-tests and statistical hypotheses."""
    df = pd.read_csv("statistical_results.csv") 
    return df

# Helper function for data visualization
def create_trend_plot(df, features):
    """Creates a multi-feature line plot using Plotly."""
    df_plot = df.melt(id_vars='Year', value_vars=features, var_name='Feature', value_name='Average Value')
    fig = px.line(
        df_plot,
        x='Year',
        y='Average Value',
        color='Feature',
        title='Temporal Trend of Spotify Audio Features',
        labels={'Average Value': 'Average Feature Value'},
        height=500
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Average Value")
    return fig

# --- Shared Data Loading for All Pages ---

def load_all_data():
    """Loads all necessary dataframes and returns them in a dict."""
    return {
        'temporal': load_temporal_data(),
        'genres': load_genre_data(),
        'decades': load_decadal_data(),
        'top_artists': load_top_artists_features(),
        'correlations': load_popularity_correlations(),
        'clusters': load_song_clusters(),
        'statistical_results': load_statistical_results()
    }
