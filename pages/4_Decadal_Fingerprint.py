import streamlit as st
import plotly.graph_objects as go
from utils import load_all_data

data = load_all_data()
df_decades = data['decades']

st.title("🔍 Decadal Hit Song Fingerprint (Q5)")
st.header("What defines the average popular song across the decades?")

# --- NEW SECTION: SUMMARY OF DECADAL CHANGE ---
st.markdown("""
### 💡 Summary of Decadal Change (1950s to 2020s)

The analysis of popular songs shows a fundamental shift in listener preference over time:

* **Louder & Energetic:** There is a clear trend toward **higher average Energy** and **Loudness** ($\approx+3.3$ dB), reflecting increased digital production and the impact of the "loudness war."
* **Less Acoustic:** **Acousticness has dropped dramatically** ($\approx-0.49$), indicating a move away from natural/live sounds toward synthesized or electronically processed music.
* **More Danceable, Less Happy:** Songs are **more Danceable** ($\approx+0.08$) but notably **less Positive (lower Valence)** ($\approx-0.22$). People are increasingly choosing tracks that are built for movement but carry a more serious or aggressive emotional tone.
""")
st.markdown("---")
# --- END NEW SECTION ---


# Define the features to show on the radar chart for Q5
q5_features = ['Danceability', 'Energy', 'Valence', 'Loudness', 'Acousticness']
decades_list = df_decades['Decade_Label'].unique().tolist()

selected_decade = st.selectbox("Select a Decade:", decades_list, index=len(decades_list)-1)

if selected_decade:
    # Filter the data for the selected decade
    df_decade_data = df_decades[df_decades['Decade_Label'] == selected_decade].iloc[0]

    r_values = []
    
    for feature in q5_features:
        value = df_decade_data[feature]
        
        # Manual normalization for display consistency
        if feature == 'Loudness':
            # Loudness is roughly -60 to 0, so normalize to 0-1 range
            norm_value = (value + 60) / 60
        else:
            # Features already scaled from 0-1
            norm_value = value
            
        r_values.append(norm_value)

    fig_decadal_radar = go.Figure(data=[
        go.Scatterpolar(
            r=r_values,
            theta=q5_features,
            fill='toself',
            name=f'{selected_decade} Hit Song Profile'
        )
    ])

    fig_decadal_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title=f"Normalized Feature Profile of Average Hit Song in the {selected_decade}"
    )
    
    st.plotly_chart(fig_decadal_radar, use_container_width=True)
    
    st.markdown("#### Key Feature Values (Un-normalized)")
    df_table = df_decade_data[q5_features].to_frame().T
    df_table.index = [f'Avg. Features in {selected_decade}']
    st.table(df_table)

    st.caption("observing the shifts in the feature fingerprint (Valence, Energy, Danceability) over the decades.")