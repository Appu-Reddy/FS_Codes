# A machine learning system was trained on a Reference Distribution of a numerical feature.

# After deployment, a new batch of data arrives called the Live Distribution.

# To measure whether the system is experiencing distribution drift, 
# we compute the:   DKL(P∥Q)

# Where:
# P = Reference distribution
# Q = Live distribution
# KL divergence measures how much information is lost when Q approximates P

# For formula: Refer to the HINT.

# Since we are given raw numeric data, we:
#     - Convert data into histograms (discrete bins)
#     - Normalize into probability distributions
#     - Apply smoothing (Laplace correction) to avoid log(0)

# Classification Rules
# --------------------
# Let:    KL = computed divergence value

# Then:
# ----------------------------------
# KL Value	        Classification
# ----------------------------------
# KL < 0.05	        Stable
# 0.05 ≤ KL < 0.25	Slight Drift
# KL ≥ 0.25       	Major Drift

# Additionally:
#     - If the number of detected peaks differs → Bimodal Shift (highest priority)

# Input Format
# ------------
# n
# <space separated n reference values>
# m
# <space separated m live values>

# Output Format
# -------------
# Print exactly one:
#     Stable
#     Slight Drift
#     Major Drift
#     Bimodal Shift


# Constraints:
# ------------
# 5 ≤ n, m ≤ 10^5
# Values are real numbers


# Sample Input-1:
# ---------------
# 6
# 10 11 12 11 10 12
# 6
# 10 11 11 12 10 12

# Sample Output-1:
# ----------------
# Stable

# Explanation:
# ------------
# KL ≈ 0.01


# Sample Input-2:
# ---------------
# 6
# 10 11 12 11 10 12
# 6
# 11 11 12 12 11 10

# Sample Output-2:
# ---------------
# Slight Drift

# Explanation:
# ------------
# KL ≈ 0.10


# Sample Input-3:
# ---------------
# 6
# 10 11 12 11 10 12
# 6
# 13 14 12 13 14 12

# Sample Output-3:
# ---------------
# Major Drift

# Explanation:
# ------------
# KL large (distributions non-overlapping)


# Sample Input-4:
# ---------------
# 6
# 10 11 12 11 10 12
# 6
# 25 26 24 25 27 26

# Sample Output-4:
# ---------------
# Bimodal Shift

# Explanation:
# ------------
# Peak count changed.

import math
import numpy as np

# ---------- Peak Detection ----------
def count_peaks(data, bins=20):
    # Implement your code here
    min_val = min(data)
    max_val = max(data)
    
    if min_val==max_val:
        return 1
        
    width = (max_val-min_val)/bins;
    
    hist = [0 for _ in range(bins)]
    
    for val in data:
        index = math.floor(((val-min_val)/width))
        if index==bins:
            index = bins - 1;
        hist[index] += 1;
        
    peaks = 0
    for i in range(1,bins-2):
        if hist[i]>hist[i-1] and hist[i]>hist[i+1]:
            peaks += 1
            
    return max(peaks,1)

# ---------- KL Divergence ----------
def kl_divergence(data1, data2, bins=20):
    # Implement your code here
    min_val = min(data1+data2)
    max_val = max(data1+data2)
    
    hist1 = [0 for _ in range(bins)]
    hist2 = [0 for _ in range(bins)]
    
    width = (max_val-min_val)/bins
    
    for x in data1:
        idx = math.floor((x-min_val)/width)
        if idx==bins:
            idx = bins-1
        hist1[idx] += 1
        
    for x in data2:
        idx = math.floor((x-min_val)/width)
        if idx==bins:
            idx = bins-1
        hist2[idx] += 1
    
    div = 0
    n = len(data1)
    m = len(data2)
    for i in range(bins):
        p = hist1[i] / n
        q = hist2[i] / m
        if q!=0:
            div += p*(np.log(p/q))
    return div
    

# ---------- Main ----------
n = int(input().strip())
reference = list(map(float, input().split()))

m = int(input().strip())
live = list(map(float, input().split()))

# Compute KL divergence
kl_score = kl_divergence(reference, live)

# Detect modality
peaks_ref = count_peaks(reference)
peaks_live = count_peaks(live)

# Classification
if peaks_ref != peaks_live:
    print("Bimodal Shift")

elif kl_score < 0.05:
    print("Stable")

elif kl_score < 0.25:
    print("Slight Drift")

else:
    print("Major Drift")
