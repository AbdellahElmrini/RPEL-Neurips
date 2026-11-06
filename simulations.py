import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from scipy.stats import hypergeom
from tqdm import tqdm, trange
import os
import seaborn as sns
sns.set_theme()


def KL_ber(alpha,beta):
    # KL divergence of Bernoulli variables
    if alpha == 0:
        return beta==0
    if alpha==1:
        return b==1
    return alpha * np.log(alpha/beta) + (1-alpha)*np.log((1-alpha)/(1-beta))

def sat(n,b,s,b_hat,T,p):
    # Returns true if the theoretical condition is satisfied 
    D = KL_ber(b_hat/s, b/(n-1))
    if D==0:
        return s==n
    return s >= min(1/D * np.log((n-b)*T/(1-p)) , n-1)


def n_th(n,b,T,p):
    # Returns the theoretical number of samples needed to satisfy Lemma 4.2

    return math.ceil( max(1/(1/2 - (b/n))**2, 3/(b/n) ) * np.log(4*(n-b)*T/ (1-p))) +2
    #return math.ceil( 1/ KL_ber(1/2, b/n) * np.log(4*n*T/ (1-p))) +2

def simulate(n,b,s, T, mult = 1):
    rng = np.random.default_rng()
    return max(rng.hypergeometric(b, n-b, s, n*T*mult))
    #return max(np.random.Generator.hypergeometric(b, n-b, s, n*T*mult))
    #return max(np.random.hypergeometric(b, n-b, s, n*T*mult))


def max_attackers_quantile(n, b, s, T, confidence=0.99):
    N = n - 1
    K = b
    draws = s

    K_total = (n - b) * T

    single_draw_confidence = confidence ** (1 / K_total)

    return int(hypergeom.ppf(single_draw_confidence, N, K, draws))




n = 100000

T = 200
s_max = 250
s_min = 30

markers = ['o', 's', '^', 'D', 'v', 'p']
s_range = range(s_min, s_max, 30)
for idx, b in enumerate(trange(6000, 32000, 4000)): # enumerate(trange(100, 320, 40)): #  trange(20, 120, 20):
    max_fraction = []
    for s in s_range:


        b_hat = max_attackers_quantile(n, b, s, T)
        max_fraction.append(b_hat / (s+1))
    
    plt.plot(s_range, max_fraction, label=f"b={b}", 
                marker=markers[idx % len(markers)], markersize=8, linewidth=2)




plt.xlabel("Number of neighbors", fontsize=12)
plt.ylabel(r"$\frac{\hat{b}}{s+1}$", rotation=0, fontsize=12)
plt.ylim(0, 1)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, alpha=0.7)
plt.legend(fontsize=10)
plt.title(rf"$n = {n}$", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create directory if it doesn't exist

dir_name = "exp_b_hat_neurips"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

plt.savefig(f"{dir_name}/b_hat_n_{n}_high_byz.png")