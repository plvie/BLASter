from math import pi, log, exp
from fpylll import Pruning, util
from scipy.special import loggamma


proba_goal = .25
overhead = 2


def slope(n):
    if n < 40:
        return -log(2.0**(-0.040))
    lV = log(pi)*(n/2) - loggamma(1+n/2)
    return -(lV + log(proba_goal/2)) * (2/n) * (1. / (n - 1))


for beta in range(2, 82, 2):
    sl = slope(beta - 2)
    profile = [exp(- sl * 2 * i) for i in range(beta)]
    rad = 1.11 * util.gaussian_heuristic(profile)
    while True:
        pr = Pruning.run(rad, 2.0**overhead, [profile], proba_goal, flags=Pruning.ZEALOUS)
        if pr.expectation > proba_goal / 1.1:
            break
        overhead += 1

    coeffs = ", ".join([f"{x:.4f}" for x in pr.coefficients])
    print(f"{{{coeffs}}}, // BKZ-{beta}, p={pr.expectation:.6f}")
