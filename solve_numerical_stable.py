import numpy as np
from scipy.linalg import solve_triangular, norm

rng = np.random.default_rng(0)
u = np.finfo(np.float64).eps/2  # unit roundoff

def gamma_n(n, u=u):
    nu = n * u
    return nu / (1.0 - nu)

def round_half_away_from_zero(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def make_upper_triangular(n, kappa_target=1e6, offdiag_scale=0.1):
    # Diagonale géométrique pour approx. contrôler κ
    d = np.logspace(0, np.log10(kappa_target), n)
    U = np.triu(rng.standard_normal((n, n)))
    np.fill_diagonal(U, 0.0)
    T = np.diag(d) + offdiag_scale * U
    return T

def method_A(T, S):  # solve_triangular
    return -solve_triangular(T, S, lower=False, check_finite=False) # the more numerically stable way

def method_B(T, S):  # np.linalg.solve
    return -np.linalg.solve(T, S) # nearly the same as method A

def method_C(T, S):  # inv @ S
    return -np.linalg.inv(T) @ S # previous method, less stable

def residual(T, X, S, p=np.inf):
    # résiduel normalisé classique
    num = norm(T @ X + S, ord=p)
    den = norm(T, ord=p) * norm(X, ord=p) + norm(S, ord=p)
    return num / den

def test_once(n=128, m=64, kappa_target=1e8):
    T = make_upper_triangular(n, kappa_target=kappa_target, offdiag_scale=0.1)
    S = rng.standard_normal((n, m))

    # Référence: solve_triangular (A)
    XA = method_A(T, S)

    # Autres méthodes
    XB = method_B(T, S)
    XC = method_C(T, S)

    # Cond. & gamma
    kappa_inf = np.linalg.cond(T, p=np.inf)
    g = gamma_n(n)

    # Forward error vs A (norm-inf)  — A sert de proxy de "vérité"
    fe_B = norm(XB - XA, ord=np.inf) / max(norm(XA, ord=np.inf), 1e-300)
    fe_C = norm(XC - XA, ord=np.inf) / max(norm(XA, ord=np.inf), 1e-300)

    # Résiduels
    rA = residual(T, XA, S)
    rB = residual(T, XB, S)
    rC = residual(T, XC, S)

    # Impact arrondi entier
    UA = round_half_away_from_zero(XA).astype(np.int64)
    UB = round_half_away_from_zero(XB).astype(np.int64)
    UC = round_half_away_from_zero(XC).astype(np.int64)

    mism_B = int(np.count_nonzero(UB != UA))
    mism_C = int(np.count_nonzero(UC != UA))

    return {
        "n": n, "m": m, "kappa_inf": kappa_inf, "gamma_n": g,
        "fe_B_vs_A": fe_B, "fe_C_vs_A": fe_C,
        "res_A": rA, "res_B": rB, "res_C": rC,
        "mismatches_B_vs_A": mism_B, "mismatches_C_vs_A": mism_C
    }

if __name__ == "__main__":
    configs = [
        (2, 1, 1e2),
        (4, 2, 1e2),
        (16, 8, 1e2),
        (32, 16, 1e2),
        (64, 32, 1e2),
        (128, 64, 1e4),
        (128, 64, 1e8),
        (256, 64, 1e10),
        (512, 64, 1e12),
        (1024, 64, 1e14),
        (2048, 64, 1e16),
    ]
    print(f"{'n':>4} {'m':>4} {'kappa_inf':>12} {'gamma_n':>10}  "
          f"{'fe_B':>10} {'fe_C':>10}  {'resA':>10} {'resB':>10} {'resC':>10}  "
          f"{'mismB':>7} {'mismC':>7}")
    for n, m, kappa in configs:
        out = test_once(n, m, kappa)
        print(f"{out['n']:4d} {out['m']:4d} "
              f"{out['kappa_inf']:12.3e} {out['gamma_n']:10.3e}  "
              f"{out['fe_B_vs_A']:10.3e} {out['fe_C_vs_A']:10.3e}  "
              f"{out['res_A']:10.3e} {out['res_B']:10.3e} {out['res_C']:10.3e}  "
              f"{out['mismatches_B_vs_A']:7d} {out['mismatches_C_vs_A']:7d}")
