from fpylll import *
from fpylll.util import gaussian_heuristic
from scipy.special import loggamma
from math import pi, log, exp


proba_goal = .25
overhead = 2


def lV(n):
    return log(pi)*(n/2) - loggamma(1+n/2)


def slope(n):
    global proba_goal

    if n < 40:
        return -log(2.0**(-0.040))
    return -(lV(n) + log(proba_goal/2)) * (2/n) * (1. / (n - 1))


for beta in range(2, 82, 2):
    # print(beta, -slope(beta)/log(2))
    # continue

    sl = slope(beta - 2)
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
