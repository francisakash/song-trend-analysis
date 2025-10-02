import streamlit as st
import pandas as pd
from utils import load_all_data

# Load dataframes from cache
data = load_all_data()
df_stats = data['statistical_results']

st.title("✅ Statistical Testing (Hypotheses)")
st.header("Are These Trends Statistically Significant?")

st.markdown("""
This section uses **statistical hypothesis testing (T-tests)** to rigorously evaluate two specific, common hypotheses about music trends and preferences. 
We test the null hypothesis ($H_0$): that there is no difference between the two groups.
""")

# --- Hypothesis 1: Danceability Over Time ---
st.markdown("---")
st.subheader("Hypothesis 1: Danceability Shift (Pre- vs. Post-2016)")

# Extract results for the first test
test1 = df_stats.iloc[0]

col_dance_1, col_dance_2 = st.columns([1, 1])

with col_dance_1:
    st.metric(
        "Post-2016 Average Danceability", 
        test1['Group_A'].split(': ')[1].replace(')', '')
    )
    st.metric(
        "Pre-2016 Average Danceability", 
        test1['Group_B'].split(': ')[1].replace(')', '')
    )
    st.metric("Average Difference", f"{test1['Mean_Diff']:.4f}")

with col_dance_2:
    st.metric("T-Statistic", f"{test1['T_Statistic']:.4f}")
    st.metric("P-Value ($\mathbf{\\alpha=0.05}$)", f"{test1['P_Value']:.10f}", delta_color="off")
    
    # Conclusion formatting
    if test1['Conclusion'].startswith('Reject'):
        st.success(f"Conclusion: {test1['Conclusion']}")
    else:
        st.warning(f"Conclusion: {test1['Conclusion']}")

st.markdown(f"""
The P-value is **{test1['P_Value']:.10f}**, which is drastically lower than the significance level ($\alpha=0.05$).

**Result:** We **reject the null hypothesis**. The data strongly supports the conclusion that songs released after 2015 (starting 2016) are **significantly more danceable** than songs released before. This confirms a real, measurable shift in the composition of popular music over the last decade.
""")

# --- Hypothesis 2: Popularity by Genre ---
st.markdown("---")
st.subheader("Hypothesis 2: Popularity Difference (Pop vs. Rock)")

# Extract results for the second test
test2 = df_stats.iloc[1]

col_pop_1, col_pop_2 = st.columns([1, 1])

with col_pop_1:
    st.metric(
        "Average Pop Popularity", 
        test2['Group_A'].split(': ')[1].replace(')', '')
    )
    st.metric(
        "Average Rock Popularity", 
        test2['Group_B'].split(': ')[1].replace(')', '')
    )
    st.metric("Average Difference", f"{test2['Mean_Diff']:.4f}")

with col_pop_2:
    st.metric("T-Statistic", f"{test2['T_Statistic']:.4f}")
    st.metric("P-Value ($\mathbf{\\alpha=0.05}$)", f"{test2['P_Value']:.10f}", delta_color="off")

    # Conclusion formatting
    if test2['Conclusion'].startswith('Reject'):
        st.success(f"Conclusion: {test2['Conclusion']}")
    else:
        st.warning(f"Conclusion: {test2['Conclusion']}")

st.markdown(f"""
The P-value is **{test2['P_Value']:.10f}**, which is significantly lower than the significance level ($\alpha=0.05$).

**Result:** We **reject the null hypothesis**. The data confirms that Pop songs are **significantly more popular** than Rock songs in the aggregate dataset. The difference in average popularity is approximately **{test2['Mean_Diff']:.2f} points**.
""")
