import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_all_data, create_trend_plot

# --- Data Loading (Full Dataset for Popularity Analysis) ---
@st.cache_data
def load_full_hits_data():
    """Loads the main, large dataset for detailed analysis (e.g., temporal analysis per artist)."""
    try:
        # NOTE: Using the accessible file name
        df = pd.read_csv("top_10000_1950-now.csv")
        # Rename popularity column for easier use
        if 'Popularity' in df.columns:
            df.rename(columns={'Popularity': 'Popularity_Score'}, inplace=True)
        
        # Ensure 'Popularity_Score' is available for comparison
        if 'Popularity_Score' not in df.columns:
            st.error("Error: 'Popularity' column not found in raw data for comparison.")
            return pd.DataFrame()

        # Convert feature columns to numeric, coercing errors to NaN
        feature_columns = ['Danceability', 'Energy', 'Valence', 'Loudness', 'Acousticness', 'Liveness', 'Tempo', 'Speechiness', 'Instrumentalness']
        for col in feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Popularity_Score'] + feature_columns)
    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot perform popularity comparison.")
        return pd.DataFrame()

# Load all core cached data
data = load_all_data()
df_temporal = data['temporal']
df_full_hits = load_full_hits_data()

st.title("🕰️ Temporal Trends & Feature Distribution")
st.header("Musical Evolution Over Time (1950-Present)")

# --- Temporal Trends Section (Original) ---
st.subheader("1. Feature Trends Over Time")

# User selects features to visualize
temporal_features = [
    'Valence', 'Energy', 'Danceability', 
    'Tempo', 'Loudness', 'Acousticness', 'Duration_ms'
]

st.markdown("#### Select Features to Plot:")
selected_trends = st.multiselect(
    "Select up to 4 features to compare their trends over time (0-1 scaled features are compared directly):",
    options=temporal_features,
    default=['Valence', 'Danceability', 'Energy']
)

if selected_trends:
    scaled_features = ['Valence', 'Energy', 'Danceability', 'Acousticness']
    non_scaled_features = ['Loudness', 'Tempo', 'Duration_ms']
    
    # Plot 0-1 scaled features
    plot_scaled_features = [f for f in selected_trends if f in scaled_features]
    if plot_scaled_features:
        st.subheader(f"Trends for Normalized Features (0-1 Scale)")
        st.plotly_chart(create_trend_plot(df_temporal, plot_scaled_features), use_container_width=True)
        st.caption("These features are normalized (0 to 1). Observe trends like the shift in **Acousticness** (natural sound) or **Valence** (mood/positiveness).")

    # Plot non-0-1 scaled features
    plot_non_scaled_features = [f for f in selected_trends if f in non_scaled_features]
    if plot_non_scaled_features:
        st.subheader(f"Trends for Specific Scale Features")
        # Plot each non-scaled feature separately for proper Y-axis scaling
        for feature in plot_non_scaled_features:
            fig = px.line(df_temporal, x='Year', y=feature, title=f"Trend of Average {feature} Over Time", height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("These features use specific scales, such as **Loudness** in dB, and show changes in production style and track structure.")

st.markdown("---")

# --- New Section: Popularity Comparison (Top 10% vs Bottom 10%) ---
st.subheader("2. Popularity Feature Distribution Comparison")
st.markdown("Compare the musical features of the **Top 10% Most Popular** songs against the **Bottom 10% Least Popular** songs using violin plots to show distribution.")

if not df_full_hits.empty:
    
    # 1. Calculate percentiles for 'Popularity_Score'
    p10 = df_full_hits['Popularity_Score'].quantile(0.10)
    p90 = df_full_hits['Popularity_Score'].quantile(0.90)

    # 2. Filter data into two groups
    df_top_10 = df_full_hits[df_full_hits['Popularity_Score'] >= p90].copy()
    df_bottom_10 = df_full_hits[df_full_hits['Popularity_Score'] <= p10].copy()

    # Add a categorical column to identify the group
    df_top_10['Popularity_Group'] = 'Top 10% Popular (N=' + str(len(df_top_10)) + ')'
    df_bottom_10['Popularity_Group'] = 'Bottom 10% Popular (N=' + str(len(df_bottom_10)) + ')'

    df_comparison = pd.concat([df_top_10, df_bottom_10])

    # 3. Feature Selection for Comparison
    comparison_features = ['Danceability', 'Energy', 'Valence', 'Loudness', 'Acousticness']
    
    selected_comparison = st.selectbox(
        "Select Feature for Distribution Comparison:",
        options=comparison_features,
        index=0
    )

    # 4. Create the Violin Plot
    if selected_comparison:
        # Set titles based on the selected feature
        title = f"Distribution of {selected_comparison} in Top 10% vs. Bottom 10% Popular Songs"
        
        # Determine the range for consistent visualization
        y_range = [0, 1]
        if selected_comparison == 'Loudness':
             # Loudness is typically between -60 and 0 dB
            y_range = [-60, 0]
        
        fig_violin = px.violin(
            df_comparison,
            x='Popularity_Group',
            y=selected_comparison,
            color='Popularity_Group',
            box=True, # Show quartiles
            points=False, # Don't plot individual points
            title=title,
            color_discrete_map={
                df_top_10['Popularity_Group'].iloc[0]: 'rgb(30, 144, 255)',  # Blue for Top
                df_bottom_10['Popularity_Group'].iloc[0]: 'rgb(255, 69, 0)' # Red for Bottom
            }
        )
        
        fig_violin.update_yaxes(range=y_range, title=selected_comparison)
        fig_violin.update_xaxes(title="")

        st.plotly_chart(fig_violin, use_container_width=True)
        
        st.caption(f"""
            This violin plot shows the range and density of **{selected_comparison}** scores for the two popularity extremes. 
            The wider the plot, the more songs exist at that value. The box inside shows the median (white line) and the interquartile range (IQR).
        """)

else:
    st.warning("Skipping Popularity Feature Distribution due to missing or invalid raw data.")
