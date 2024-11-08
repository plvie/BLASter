from fpylll import *


# As = [IntegerMatrix.from_file(f"../output/d4_256_829561_{x}") for x in range(10)]
# As = [IntegerMatrix.from_file(f"../output/d4_128_631_{x}") for x in range(10)]
# 
# Ms = [GSO.Mat(A) for A in As]
# assert all([M.update_gso() for M in Ms])
# offset = Ms[0].d // 2
# profs = [M.r()[offset:] for M in Ms]
# renorm = sum(sum(prof) for prof in profs) / sum(len(prof) for prof in profs)
# for prof in profs:
    # prof = [p / renorm for p in prof]

profs = [[2.0**(-0.044128 * 2 * i) for i in range(64)]]
radius = sum([prof[0] for prof in profs]) / len(profs)

FPLLL.set_precision(256)
# for beta in range(2, 65, 2):
for beta in [52]:
    try:
        profs_ = [prof[:beta] for prof in profs]
        pr = Pruning.run(radius, 2.0**30, profs_, 0.8, flags=Pruning.ZEALOUS, float_type='mpfr')
        coeffs = ",".join([f"{x:.3g}" for x in pr.coefficients])
        print("\t{", coeffs, f"}}, // BKZ-{beta}, p={pr.expectation:.6f}", sep="")
    except RuntimeError:
        print(f"Beta={beta} failed.")
