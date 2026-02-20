# Deep networks suffer from internal covariate shift, where layer inputs change 
# distribution during training. Batch Normalization stabilizes training by 
# normalizing activations per feature/channel and then applying learnable scaling 
# and shifting 

# You must implement the forward pass of Batch Normalization for:
#     - 2D input: shape (N, D)
#     - 4D input: shape (N, C, H, W)

# Mathematical Definition: Refer the HINT.

# 📥 Input Format
# ---------------
# If 2D:
# 2D
# N D
# <next N lines each with D values>
# <gamma (D values)>
# <beta (D values)>
# eps

# if 4D:
# 4D
# N C H W
# <next line: flattened N*C*H*W values>
# <gamma (C values)>
# <beta (C values)>
# eps

# 📤 Output
# ---------
# Return normalized tensor y with same shape as x.

# Sample Input-1:
# -------------
# 2D
# 3 2
# 1 2
# 3 6
# 5 10
# 1 0.5
# 0 1
# 1e-5

# Sample Output-1:
# ----------------
# -1.2247 0.3876
# 0.0000 1.0000
# 1.2247 1.6124

# Explanation: Refer to the HINT


# Sample Input-2:
# ---------------
# 4D
# 2 2 1 1
# 1 2 3 4
# 1 0.5
# 0 -1
# 0

# Sample Output:
# --------------
# -1.0000
# -1.5000
# 1.0000
# -0.5000

# Explanation: Refer to the HINT

import numpy as np

def batchnorm_2d():
    N, D = map(int, input().split())
    
    x = []
    for _ in range(N):
        x.append(list(map(float, input().split())))
    x = np.array(x)
    
    gamma = np.array(list(map(float, input().split())))
    beta = np.array(list(map(float, input().split())))
    eps = float(input())
    # Compute per-feature (column-wise) mean and variance across the batch, 
    # normalize, then scale and shift using gamma and beta.
    
    #IMPLEMENT YOUR CODE HERE
    mean = np.mean(x,axis=0)
    var = np.var(x,axis=0)
    x2 = (x-mean)/np.sqrt(var_eps)
    y = gamma*x2 + beta
    for r in y:
        print("".join())


def batchnorm_4d():
    N, C, H, W = map(int, input().split())
    
    total = N * C * H * W
    values = list(map(float, input().split()))
    
    if len(values) != total:
        raise ValueError("Incorrect number of tensor values provided.")
    
    x = np.array(values).reshape(N, C, H, W)
    
    gamma = np.array(list(map(float, input().split())))
    beta = np.array(list(map(float, input().split())))
    eps = float(input())
    
    # Compute per-channel mean and variance across (N, H, W), normalize, 
    # then broadcast gamma and beta to scale and shift each channel.
    
    #IMPLEMENT YOUR CODE HERE
    mean = np.mean(x,axis=[0,2,3])
    var = np.var(x,axis=[0,2,3])

# MAIN METHOD
def main():
    mode = input().strip()
    
    if mode == "2D":
        batchnorm_2d()
    elif mode == "4D":
        batchnorm_4d()
    else:
        print("Invalid mode. Use '2D' or '4D'.")


if __name__ == "__main__":
    main()