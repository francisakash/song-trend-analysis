import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_all_data

# --- Data Loading (Full Dataset for Detailed Exploration) ---
@st.cache_data
def load_exploratory_data():
    """Loads and preprocesses the full dataset for flexible time, genre, and artist analysis."""
    try:
        df = pd.read_csv("top_10000_1950-now.csv")
        
        # Standardize column names for processing
        df.rename(columns={
            'Album Release Date': 'Release_Date',
            'Artist Name(s)': 'Artist_Name',
            'Track Name': 'Track_Name',
            'Danceability': 'Danceability', 
            'Energy': 'Energy', 
            'Valence': 'Valence', 
            'Acousticness': 'Acousticness', 
            'Loudness': 'Loudness',
            'Tempo': 'Tempo',
            'Artist Genres': 'Artist_Genres'
        }, inplace=True)

        # 1. Extract Year
        df['Year'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year
        df = df.dropna(subset=['Year']).astype({'Year': 'int'})
        
        # 2. Convert features to numeric (coercing errors)
        feature_columns = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Loudness', 'Tempo']
        for col in feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna(subset=feature_columns)
    
    except FileNotFoundError:
        st.error("Error: 'top_10000_1950-now.csv' not found. Cannot perform exploratory analysis.")
        return pd.DataFrame()

# Load data
df_raw = load_exploratory_data()

st.title("🧭 Exploratory Trends Dashboard")
st.header("Analyze Feature Averages by Year, Genre, or Artist")
st.markdown("""
Use this flexible tool to see how a specific musical feature has trended when grouped by time, genre, or artist.
""")

if df_raw.empty:
    st.stop()

# --- Control Panel ---
# 1. Feature Selection
feature_list = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Loudness', 'Tempo']
# FIX: Replaced 'default' with 'index'. 'Energy' is the second item (index 1).
selected_feature = st.selectbox(
    "1. Select Audio Feature to Track:", 
    options=feature_list,
    index=1 
)

# 2. Grouping Dimension Selection
grouping_dimension = st.radio(
    "2. Group Data By:", 
    options=['Year', 'Artist_Name', 'Artist_Genres'],
    format_func=lambda x: x.replace('_', ' '),
    horizontal=True,
    index=0
)

# --- Dynamic Analysis ---

if grouping_dimension == 'Year':
    # Group by Year
    df_grouped = df_raw.groupby('Year')[selected_feature].mean().reset_index()
    chart_type = 'line'
    x_axis_title = 'Release Year'
    df_plot = df_grouped.sort_values('Year')
    title_suffix = "Over Time"
    # Column used in plot is 'Year'
    plot_dimension = 'Year'

elif grouping_dimension == 'Artist_Name':
    # Group by Artist (Filter to top N artists for performance and clarity)
    # Filter to artists with at least 5 hit songs
    artist_counts = df_raw['Artist_Name'].value_counts()
    top_artists = artist_counts[artist_counts >= 5].index.tolist()
    
    if not top_artists:
        st.warning("No artists with 5 or more songs found to display.")
        st.stop()
        
    df_filtered_artists = df_raw[df_raw['Artist_Name'].isin(top_artists)]
    
    # Calculate the average feature score per artist
    df_grouped = df_filtered_artists.groupby('Artist_Name')[selected_feature].mean().reset_index()
    
    # Show top 20
    df_plot = df_grouped.sort_values(selected_feature, ascending=False).head(20)
    chart_type = 'bar'
    x_axis_title = 'Artist Name'
    title_suffix = "for Top 20 Prolific Artists"
    # Column used in plot is 'Artist_Name'
    plot_dimension = 'Artist_Name'

elif grouping_dimension == 'Artist_Genres':
    # Group by primary Artist Genre (requires splitting the comma-separated list)
    
    # Explode the genres into separate rows
    df_exploded = df_raw.assign(Genre=df_raw['Artist_Genres'].str.split(',')).explode('Genre')
    df_exploded['Genre'] = df_exploded['Genre'].str.strip()
    
    # Filter out empty or unclassified genres
    df_exploded = df_exploded[df_exploded['Genre'].str.len() > 0]

    # Calculate the average feature score per genre
    df_grouped = df_exploded.groupby('Genre')[selected_feature].mean().reset_index()
    
    # Only show genres with a minimum number of songs (e.g., 20)
    genre_counts = df_exploded['Genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 20].index
    
    df_plot = df_grouped[df_grouped['Genre'].isin(valid_genres)]
    
    # Show top 20 genres
    df_plot = df_plot.sort_values(selected_feature, ascending=False).head(20)
    chart_type = 'bar'
    x_axis_title = 'Genre'
    title_suffix = "for Top 20 Found Genres"
    # Column used in plot is 'Genre'
    plot_dimension = 'Genre'


# --- Visualization ---
if not df_plot.empty:
    st.subheader(f"Average {selected_feature} {title_suffix}")
    
    y_axis_title = f"Average {selected_feature}"
    
    # The dimension used on the x/y axis changes based on the grouping
    # For bar charts, the dimension is on the Y-axis. For the line chart, it's on the X-axis.

    if chart_type == 'line':
        fig = px.line(
            df_plot, 
            x=plot_dimension, 
            y=selected_feature, 
            title=f"Annual Trend of {selected_feature}",
            markers=True
        )
    else: # Bar chart for Genre and Artist
        fig = px.bar(
            df_plot, 
            x=selected_feature, 
            y=plot_dimension, # <-- FIXED: Uses plot_dimension (either 'Artist_Name' or 'Genre')
            orientation='h',
            title=f"Comparison of {selected_feature}",
            color=selected_feature,
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={selected_feature: y_axis_title, plot_dimension: x_axis_title}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Insufficient data to generate the trend plot for the selected options.")
