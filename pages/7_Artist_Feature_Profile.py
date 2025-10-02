import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_all_data # Assuming this loads cached data structures

# --- Data Loading ---
# We no longer need the large, full hits data, only the aggregated features data.

# Load all core cached data
# Assuming load_all_data() returns {'top_artists': df_top_artists_features}
try:
    data = load_all_data()
    df_top_artists_features = data.get('top_artists', pd.DataFrame())
except Exception as e:
    st.error(f"Failed to load required artist features data: {e}")
    df_top_artists_features = pd.DataFrame()


# --- Utility Functions for Display ---

def display_chart_dominance(df):
    """Generates and displays the chart dominance bar chart."""
    st.subheader("1. Chart Dominance: The Most Prolific Artists")

    st.markdown("""
    This chart shows the **number of hit songs** (`Song_Count`) for the most successful artists over the entire 1950-Present period, 
    measuring their overall **dominance** and sustained charting success.
    """)

    COUNT_COLUMN = 'Song_Count' 
    
    if df.empty or COUNT_COLUMN not in df.columns:
        st.warning("Artist features data not available for dominance chart.")
        return

    # Filter and sort data
    df_chart = df.sort_values(COUNT_COLUMN, ascending=False).head(20)

    fig_dominance = px.bar(
        df_chart,
        x=COUNT_COLUMN,
        y='Artist_Name',
        orientation='h',
        title="Top 20 Artists by Number of Hit Songs (1950-Present)",
        labels={COUNT_COLUMN: 'Total Hit Count', 'Artist_Name': 'Artist'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_dominance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_dominance, use_container_width=True)


# --- Main Page Execution ---
st.title("🎤 Top Artist Feature Profile")
st.header("Analyzing Chart Dominance")

# 1. Chart Dominance (The only remaining feature)
display_chart_dominance(df_top_artists_features)
