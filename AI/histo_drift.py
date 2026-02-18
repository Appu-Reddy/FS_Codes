# A machine learning system was trained using a historical dataset called the 
# Reference Window. After deployment, new incoming data forms a Live Window.

# To ensure model reliability, we must detect whether the Live Window distribution 
# has shifted relative to the Reference Window.

# Your task is to classify the type of shift.


# When comparing two numeric datasets, distribution shift can be measured using:
#     - Mean Shift (Location Drift): Measures change in central tendency.
    
#     - Variance Shift (Spread Drift)" Measures change in dispersion.
    
#     - Distribution Shape Shift (Modality Change): Detects if distribution becomes 
#                                                 bimodal (two peaks).
#     - Distribution Divergence (Histogram Distance): Measures overall difference 
#                                             using normalized histogram distance.

# Classification Rules
# --------------------
# Let:
#     Δμ = absolute mean difference
#     Δσ = absolute standard deviation difference
#     D = histogram divergence score (range 0–1)
#     ModeChange = whether number of peaks differs

# Final Classification Logic (Priority Order)
# -------------------------------------------
# If ModeChange is TRUE → Bimodal Shift
# Else if D ≥ 0.40 → Major Drift
# Else if D ≥ 0.15 → Slight Drift
# Else → Stable

# Input Format:
# -------------
# n
# <space separated n reference values>
# m
# <space separated m live values>


# Output Format:
# --------------
# Print exactly one of:
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
# Mean nearly same
# Variance similar
# Histogram overlap high
# No shape change


# Sample Input-2:
# ---------------
# 6
# 10 11 12 11 10 12
# 6
# 11 12 13 12 11 12

# Sample Output-2:
# ---------------
# Slight Drift

# Explanation:
# ------------
# Moderate shift in mean
# Histogram divergence around 0.2–0.3


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
# Large mean difference
# Almost no histogram overlap


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
# Distribution changed from unimodal to bimodal


import math
from collections import Counter

# ---------- Statistics ----------
def mean(data):
    return sum(data)/len(data)

def std_dev(data, mu):
    tot = 0
    for x in data:
        tot += (mu-x)**2
    return math.sqrt(tot/len(data))

# ---------- Histogram Divergence ----------
def histogram_divergence(data1, data2, bins=20):
    min_val = min(data1+data2)
    max_val = max(data1+data2)
    
    width = (max_val-min_val)/bins
    
    hist1 = [0 for _ in range(bins)]
    hist2 = [0 for _ in range(bins)]
    
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
    for i in range(bins):
        p = hist1[i] / n
        q = hist2[i] / m
        div += abs(p-q)
        
    return div/2
    
# ---------- Peak Detection ----------
def count_peaks(data, bins=20):
    min_val = min(data)
    max_val = max(data)
    
    if min_val==max_val:
        return 1
        
    width = (max_val-min_val)/bins
    
    histogram = [0 for _ in range(bins)]
    
    for val in data:
        index = math.floor(((val-min_val)/width))
        if index==bins:
            index = bins - 1;
        histogram[index] += 1;
        
    peaks = 0
    
    for i in range(1,bins-2):
        if histogram[i]>histogram[i-1] and histogram[i]>histogram[i+1]:
            peaks += 1
            
    return max(peaks,1)
    
# ---------- Main ----------
n = int(input().strip())
reference = list(map(float, input().split()))

m = int(input().strip())
live = list(map(float, input().split()))

# Compute statistics
mu_ref = mean(reference)
mu_live = mean(live)

std_ref = std_dev(reference, mu_ref)
std_live = std_dev(live, mu_live)

div_score = histogram_divergence(reference, live)

peaks_ref = count_peaks(reference)
peaks_live = count_peaks(live)

# ---------- Classification ----------

if peaks_ref != peaks_live:
    print("Bimodal Shift")

elif div_score >= 0.40:
    print("Major Drift")

elif div_score >= 0.15:
    print("Slight Drift")

else:
    print("Stable")
