import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import all functions from the utility file
from apputil import kmeans, kmeans_diamonds, kmeans_timer, get_bin_search_steps, DIAMONDS_NUMERIC_DF

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="Week 11: K-Means & Time Complexity")
sns.set_theme(style="whitegrid") # Set plotting style

st.title("K-Means Clustering & Algorithm Complexity Analysis")
st.markdown("This application showcases the results for the Week 11 assignment, including K-Means performance and Binary Search step-counting.")

# --- Helper Functions for generating data ---

@st.cache_data
def run_kmeans_timing_tests():
    """Generates data for Exercise 3 plot (Time vs n and Time vs k)."""

    # K-Means Time Complexity vs. n (Number of Samples)
    n_values = np.arange(100, 50000, 1000)
    k5_times = [kmeans_timer(n, 5, 20) for n in n_values]

    # K-Means Time Complexity vs. k (Number of Clusters)
    k_values = np.arange(2, 50)
    n10k_times = [kmeans_timer(10000, k, 10) for k in k_values]
    
    return n_values, k5_times, k_values, n10k_times

@st.cache_data
def run_binary_search_steps():
    """Generates data for the Bonus Exercise plot (Steps vs n)."""
    
    n_bonus_values = [2**i for i in range(1, 21)]
    step_counts = [get_bin_search_steps(n) for n in n_bonus_values]

    log_n_theory = np.log2(n_bonus_values)
    scaling_factor = step_counts[-1] / log_n_theory[-1]
    scaled_theory = log_n_theory * scaling_factor
    
    return n_bonus_values, step_counts, scaled_theory

# --- Data Loading (Run once on first load) ---
# We use session_state to run this block only on the first page load
# to show a nice "loading" animation.
if 'data_loaded' not in st.session_state:
    # UPDATED: Changed spinner text and icon, removed st.balloons()
    with st.spinner("⏳ Running initial data analysis and generating plots... This may take a minute."):
        # Run and cache all data
        run_kmeans_timing_tests()
        run_binary_search_steps()
        # Mark as loaded
        st.session_state.data_loaded = True
else:
    # On subsequent re-runs, just pull from cache (will be instant)
    n_values, k5_times, k_values, n10k_times = run_kmeans_timing_tests()
    n_bonus_values, step_counts, scaled_theory = run_binary_search_steps()

# --- Exercise 1 & 2: K-Means Verification ---
st.header("1. K-Means on Diamonds (Exercises 1 & 2)")
st.markdown(
    """
    **Exercise 1** (`kmeans`) provides the core clustering logic using Scikit-Learn.  
    **Exercise 2** (`kmeans_diamonds`) applies this logic to the `diamonds` dataset. 
    Below is a sample run to verify the functions are working correctly.
    """
)

# Test run parameters
n_test = 2000
k_test = 4
centroids_ex2, labels_ex2 = kmeans_diamonds(n=n_test, k=k_test)
n_features = len(DIAMONDS_NUMERIC_DF.columns)

# Display test parameters as metrics
st.subheader("Verification Test Run Parameters")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Samples Used (n)", f"{n_test:,}")
col_m2.metric("Clusters (k)", k_test)
col_m3.metric("Features (d)", n_features, help=f"Features: {', '.join(DIAMONDS_NUMERIC_DF.columns)}")

st.subheader("Sample Output")
st.markdown(
    """
    - **Centroids:** The central values for each of the $k$ clusters across all $d$ features (e.g., `carat`, `depth`, `price`).
    - **Labels:** The cluster index (0 to $k-1$) assigned to each of the $n$ samples.
    """
)

# Display output dataframes
col1, col2 = st.columns(2)
with col1:
    st.write("Cluster Centroids (First 2 rows):")
    feature_names = DIAMONDS_NUMERIC_DF.columns.tolist()
    centroids_df = pd.DataFrame(centroids_ex2, columns=feature_names)
    st.dataframe(centroids_df.head(2).style.format(precision=3))

with col2:
    st.write("First 10 Cluster Labels:")
    labels_df = pd.DataFrame(labels_ex2[:10], columns=["Cluster Label"])
    # FIX: Rename the column to a shorter name to reduce table width
    labels_df.rename(columns={"Cluster Label": "ID"}, inplace=True)
    st.dataframe(labels_df)

st.divider()

# --- Exercise 3: K-Means Time Complexity ---
st.header("2. K-Means Time Complexity (Exercise 3)")
st.markdown(
    """
    The `kmeans_timer` function measures runtime to analyze how K-Means performance scales. 
    The theoretical complexity is roughly **$O(I \cdot k \cdot n \cdot d)$** ($I$ = iterations). 
    We test this by varying $n$ and $k$ independently.
    """
)

# Pull data from the cached functions
n_values, k5_times, k_values, n10k_times = run_kmeans_timing_tests()

col3, col4 = st.columns(2)

# Plot 1: Time vs. n (Number of Rows)
with col3:
    st.subheader('Runtime vs. Number of Samples ($n$)')
    st.markdown(
        """
        This plot keeps $k=5$ constant and increases $n$. The resulting runtime
        scales **linearly** (a straight line), which matches the **$O(n)$** term
        in the complexity formula.
        """
    )
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=n_values, y=k5_times, ax=ax1)
    ax1.set_xlabel("Number of Rows (n)")
    ax1.set_ylabel("Average Time (seconds)")
    st.pyplot(fig1)

# Plot 2: Time vs. k (Number of Clusters)
with col4:
    st.subheader('Runtime vs. Number of Clusters ($k$)')
    st.markdown(
        """
        This plot keeps $n=10,000$ constant and increases $k$. The runtime also
        scales **linearly** (a straight line), matching the **$O(k)$** term in the
        complexity formula.
        """
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=k_values, y=n10k_times, ax=ax2)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Average Time (seconds)")
    st.pyplot(fig2)

st.divider()

# --- Bonus Exercise: Binary Search Time Complexity ---
st.header("3. Bonus: Binary Search Complexity")
st.markdown(
    """
    This exercise analyzes the `bin_search(n)` function not by time, but by
    counting its exact computational steps in the *worst-case scenario* to empirically determine its time complexity.
    """
)

# Pull data from the cached functions
n_bonus_values, step_counts, scaled_theory = run_binary_search_steps()

col5, col6 = st.columns([3, 2]) # Give the plot more space

with col5:
    st.subheader("Step Count vs. Input Size (n)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    
    # Plot the Empirical Steps
    sns.lineplot(x=n_bonus_values, y=step_counts, ax=ax3, marker='o', label='Empirical Steps (Worst-Case)')
    
    # Plot the Theoretical Curve (O(log n))
    sns.lineplot(x=n_bonus_values, y=scaled_theory, ax=ax3, linestyle='--', color='red', label='Theoretical $O(\log n)$ Curve')
    
    ax3.set_xlabel("Input Size (n) — Log Scale")
    ax3.set_ylabel("Number of Computational Steps")
    ax3.set_xscale('log') # A log scale on x-axis shows the relationship clearly
    ax3.set_title('Binary Search Step Complexity: $O(\log n)$')
    ax3.legend()
    st.pyplot(fig3)

with col6:
    st.subheader("Complexity Analysis")
    st.info("Estimated Time Complexity: $O(\log n)$")
    st.markdown(
        """
        **Explanation:**
        The empirical step count (blue) perfectly matches the theoretical
        **logarithmic** curve (red). This means that doubling the input size
        (e.g., from 1M to 2M) only adds a single, constant amount of work.
        """
    )
    st.write("---")
    
    st.subheader("Example Data Points")
    example_df = pd.DataFrame({
        "Input Size (n)": n_bonus_values[-5:],
        "Steps (Empirical)": step_counts[-5:],
        "Log$_2$(n)": np.log2(n_bonus_values[-5:]).astype(int)
    })
    st.dataframe(example_df.set_index("Input Size (n)").style.format(precision=0))
