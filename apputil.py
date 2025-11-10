import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time
import warnings

# Suppress warnings that might clutter the output in the terminal/app
warnings.filterwarnings('ignore')

# --- Exercise 2 Setup: Global Data Loading ---
# 1. Load the 'diamonds' dataset from seaborn.
DIAMONDS_DF = sns.load_dataset('diamonds')
# 2. Identify and save just the numerical columns as a global variable.
DIAMONDS_NUMERIC_DF = DIAMONDS_DF.select_dtypes(include=np.number)
# ---------------------------------------------

## Exercise 1: kmeans(X, k)
def kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs k-means clustering on a numerical NumPy array X.
    Returns a tuple (centroids, labels).
    """
    # Use n_init='auto' for modern sklearn compatibility and random_state=0 for reproducibility
    kmeans_model = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans_model.fit(X)

    return (kmeans_model.cluster_centers_, kmeans_model.labels_)


## Exercise 2: kmeans_diamonds(n, k)
def kmeans_diamonds(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs k-means clustering on the first 'n' rows of the numeric diamonds dataset
    to create 'k' clusters.
    """
    # 1. Select the first 'n' rows.
    data_subset = DIAMONDS_NUMERIC_DF.head(n)

    # 2. Convert the DataFrame subset to a NumPy array for the kmeans function.
    X = data_subset.values

    # 3. Run the kmeans function.
    return kmeans(X, k)


## Exercise 3: kmeans_timer(n, k, n_iter=5)
def kmeans_timer(n: int, k: int, n_iter: int = 5) -> float:
    """
    Runs kmeans_diamonds(n, k) exactly n_iter times, saves the runtime for each run,
    and returns the average time.
    """
    runtimes = []

    for _ in range(n_iter):
        start = time() # capture the starting time
        _ = kmeans_diamonds(n, k) # Run the clustering function
        t = time() - start # calculate the runtime
        runtimes.append(t)

    # Return the average time
    return np.mean(runtimes)

# --- Bonus Exercise: Binary Search Time Complexity ---

# Global variable to track the number of computational steps
step_count = 0

def bin_search(n: int) -> int:
    """
    Binary search function modified to count computational steps in the worst case.
    Target (x=n) is guaranteed to be outside the array [0, n-1].
    """
    global step_count

    # Initial setup
    arr = np.arange(n)
    left = 0
    right = n-1
    x = n 

    while left <= right:
        # Step 1: Calculate middle (arithmetic, assignment)
        middle = left + (right - left) // 2
        step_count += 1

        # Step 2: Check if x is present (comparison)
        if (arr[middle] == x):
            step_count += 1
            return middle

        # Step 3 & 4: Comparison and Assignment (2 steps)
        if (arr[middle] < x):
            left = middle + 1
            step_count += 2
        # Step 5 & 6: Comparison (implicitly in 'else') and Assignment (2 steps)
        else:
            right = middle - 1
            step_count += 2

    return -1

def get_bin_search_steps(n: int) -> int:
    """
    Runs bin_search(n) in the worst-case scenario and returns the step count,
    resetting the global step_count before running.
    """
    global step_count
    
    # Reset the counter before the run
    step_count = 0
    
    # Run the worst-case search
    _ = bin_search(n)
    
    return step_count
