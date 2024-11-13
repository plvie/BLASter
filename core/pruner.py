from fpylll import *
from fpylll.util import gaussian_heuristic
from scipy.special import loggamma
from math import pi, log, exp


proba_goal = .25
overhead = 2

def lV(n):
    return log(pi)*(n/2) - loggamma(1+n/2)

def slope(n, proba=.5):
    if n < 32:
        return log(2**.040)
    return -(lV(n) + log(proba/2))*(2/n) *(1./(1.*n-1.))

# for i in range(60):
#     print(i, slope(i))
# exit(1)


for beta in range(2, 82, 2):
    sl = slope(beta-2, proba_goal)
    profile = [exp(- sl * 2 * i) for i in range(beta)]
    rad = 1.11 * gaussian_heuristic(profile)
    while True:
        pr = Pruning.run(rad, 2.0**overhead, [profile], proba_goal, flags=Pruning.ZEALOUS)
        if pr.expectation > proba_goal / 1.1:
            break
        overhead += 1
        # print("oh = ", overhead)


    coeffs = ", ".join([f"{x:.4f}" for x in pr.coefficients])
    print("{", coeffs, f"}}, // BKZ-{beta}, p={pr.expectation:.6f}")
