import streamlit as st
import plotly.express as px
from utils import load_all_data

data = load_all_data()
df_corr = data['correlations']

st.title("🔥 Popularity Drivers")
st.header("What Audio Features are Most Associated with Popular Tracks?")

# --- Feature Correlation Bar Chart (Answering Q: What features are most associated?) ---

st.subheader("Correlation of Audio Features with Track Popularity")
st.markdown(
    """
    This chart uses the **Pearson correlation coefficient** (from -1 to +1) to measure the linear relationship between each feature and a track's Spotify popularity score.
    * **Positive values** (to the right) indicate features that tend to be **higher** in popular tracks.
    * **Negative values** (to the left) indicate features that tend to be **lower** in popular tracks.
    """
)

# Sort by coefficient for clear visualization
df_plot = df_corr.sort_values('Correlation_Coefficient', ascending=True)

# Define color scale based on coefficient value
# Positive correlations (drivers) are blue, negative (inhibitors) are red
colors = ['skyblue' if c > 0 else 'salmon' for c in df_plot['Correlation_Coefficient']]

fig_corr = px.bar(
    df_plot,
    x='Correlation_Coefficient',
    y='Feature',
    orientation='h',
    title="Feature Correlation with Popularity Score",
    labels={'Correlation_Coefficient': 'Pearson Correlation Coefficient (r)', 'Feature': 'Audio Feature'},
    color=colors,
    color_discrete_map="identity" # Use the defined colors list
)

fig_corr.update_layout(
    xaxis_range=[-0.1, 0.1], # Zoom into the small correlation range
    xaxis_title="Correlation Coefficient (r)",
    yaxis_title="Feature",
    showlegend=False
)

st.plotly_chart(fig_corr, use_container_width=True)

# --- Summary and Answer (Answering Q: Is there a strong correlation between energy and popularity?) ---

st.markdown("---")
st.subheader("Conclusion: Strength of Correlation")

# Find the correlation for Energy specifically
energy_corr = df_corr[df_corr['Feature'] == 'Energy']['Correlation_Coefficient'].iloc[0]

col_summary, col_kpi = st.columns([3, 1])

with col_summary:
    st.markdown(f"""
    1.  **Energy and Popularity:** The correlation coefficient between **Energy** and **Popularity** is **$\mathbf{{r = {energy_corr:.4f}}}$**. This value is extremely close to zero, meaning there is **no strong linear correlation** between the two. While energy *levels* have increased over time, the energy of a single track is not a primary driver of its modern popularity score.
    2.  **Key Popularity Drivers:** The features with the highest (though still weak) positive correlation are **Danceability** and **Loudness**. This suggests that popular tracks tend to be slightly easier to move to and produced at a higher volume.
    3.  **Key Popularity Inhibitors:** The features with the strongest negative correlation are **Instrumentalness**, **Liveness**, and **Acousticness**. Tracks that are perceived as heavily acoustic, recorded live, or purely instrumental are generally **less** likely to achieve a high popularity score in this dataset.
    """)

with col_kpi:
    st.metric("Energy Correlation (r)", f"{energy_corr:.4f}")
    st.metric("Strongest Positive Driver", df_corr.iloc[0]['Feature'])
    st.metric("Strongest Negative Inhibitor", df_corr.iloc[-1]['Feature'])

st.caption("Note: Since all correlations are relatively weak (close to zero), no single audio feature strongly dictates whether a song becomes popular. Popularity is driven by external factors like artist fame, marketing, and cultural timing.")
