from fpylll import *


# As = [IntegerMatrix.from_file(f"../output/d4_256_829561_{x}") for x in range(10)]
As = [IntegerMatrix.from_file(f"../output/d4_128_631_{x}") for x in range(10)]

Ms = [GSO.Mat(A) for A in As]
assert all([M.update_gso() for M in Ms])
offset = Ms[0].d // 2
radius = sum([M.get_r(offset, offset) for M in Ms]) / len(Ms)

FPLLL.set_precision(256)
for beta in range(2, 33):
    try:
        profiles = [M.r()[offset:offset + beta] for M in Ms]
        pr = Pruning.run(radius, 2.0**30, profiles, 0.8, flags=Pruning.ZEALOUS, float_type='mpfr')
        # pr = Pruning.run(radius, 2.0**30, profiles, 0.8, flags=Pruning.NELDER_MEAD, float_type='mpfr', pruning=pr0)

        coeffs = ", ".join([f"{x:.3f}" for x in pr.coefficients])
        print("\t{", coeffs, f"}}, // BKZ-{beta}, p={pr.expectation:.6f}")
    except RuntimeError:
        print(f"Beta={beta} failed.")
