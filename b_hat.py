import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from scipy.stats import hypergeom
from tqdm import tqdm, trange
import time
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

def simulate_scipy(n,b,s, T, mult = 1):
    return max(hypergeom.rvs(n-1, b, s, size=n*T*mult))

def simulate_max_scipy(n, b, s, T, mult=1, honest_only=False, size=None, rng=None):
    rng = np.random.default_rng(rng)

    N = n - 1
    K = b
    draws = s

    num_nodes = n - b if honest_only else n
    K_total = num_nodes * T * mult

    u = rng.random(size)
    q = u ** (1 / K_total)

    return hypergeom.ppf(q, N, K, draws).astype(int)

def max_attackers_quantile(n, b, s, T, confidence=0.99):
    N = n - 1
    K = b
    draws = s

    K_total = (n - b) * T

    single_draw_confidence = confidence ** (1 / K_total)

    return int(hypergeom.ppf(single_draw_confidence, N, K, draws))

n = 200
b = 20
s = 20

T = 200
print("Starting")
t= time.time()
b_hat = max_attackers_quantile(n, b, s, T, confidence=0.95)
print("Time taken:", time.time()-t)
print(b_hat)
print(b_hat/s)


