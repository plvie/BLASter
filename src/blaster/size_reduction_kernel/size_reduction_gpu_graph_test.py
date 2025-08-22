"""
Utility functions for calling Babai's nearest plane algorithm, size-reducing a basis or
Seysen-reducing a basis.

In comments, the old recursive functions are kept for clarity.
"""
from functools import cache
import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular


from ctypes import c_void_p, c_int, c_longlong, c_double, byref
# Local imports


def is_weakly_lll_reduced_gpu(R, host_flag, delta=0.99):
    """
    Fully GPU-side check for the Weak-LLL condition:
      ∀ pos: ||b_{pos+1}||^2 > δ · ||b_pos||^2
    where b_pos = (R[pos,pos], 0) and b_{pos+1} = (〈R[pos,pos+1]〉_u, R[pos+1,pos+1]).
    Returns a Python bool, but only synchronizes once at the very end.
    """
    n = R.shape[0]
    # positions 0 .. n-2
    i = cp.arange(n - 1)

    # diagonal and super-diagonal entries
    u = cp.abs(R[i,     i    ])   # shape (n-1,)
    v =        R[i,     i + 1]   # shape (n-1,)
    w =        R[i + 1, i + 1]   # shape (n-1,)

    # compute centered v_mod = (v mod u) in [−u/2, +u/2)
    v_mod = v - cp.rint(v / u) * u

    # LLL condition: v_mod**2 + w**2 > δ * u**2
    ok = (v_mod**2 + w**2) > (delta * u**2)

    # cp.all returns a 0-d GPU array; .item() brings back one host bool
    ok_scalar = cp.all(ok)   # GPU-side scalar
    ptr_dev = ok_scalar.data.ptr
    ptr_host = host_flag.ptr  # pointeur hôte
    stream = cp.cuda.Stream.null
    cp.cuda.runtime.memcpyAsync(
        ptr_host, ptr_dev,
        1,  # 1 byte – cupy bool
        cp.cuda.runtime.memcpyDeviceToHost,
        stream.ptr
    )



@cache
def __reduction_ranges(n):
    """
    Return list of ranges that needs to be reduced.

    More generally, it returns, without using recursion, the list that would be
    the output of the following Python program:

    <<<BEGIN CODE>>>
    def rec_range(n):
        bc, res = [], []
        def F(l, r):
            if l == r:
                return
            if l + 1 == r:
                bc.append(l)
            else:
                m = (l + r) // 2
                F(l, m)
                F(m, r)
                res.append((l, m, r))
        return F(0, n)
    <<<END CODE>>>

    :param n: the length of the array that requires reduction
    :return: pair containing `the base_cases` and `result`.
             `base_cases` is a list of indices `i` such that:
                `i + 1` needs to be reduced w.r.t. `i`.
             `result` is a list of triples `(i, j, k)` such that:
                `[j:k)` needs to be reduced w.r.t. `[i:j)`.
             The guarantee is that for any 0 <= i < j < n:
             1) `i in base_cases && j = i + 1`,
             OR
             2) there is a triple (u, v, w) such that `i in [u, v)` and `j in [v, w)`.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n

            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift

            if right_idx > left_idx + 1:
                # Only consider nontrivial intervals
                if right_idx == left_idx + 2:
                    # Return length 2 intervals separately to unroll base case.
                    base_cases.append(left_idx)
                else:
                    # Properly sized interval:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))



__reduction_cache = {}
_rows_h_cache = {}
_cols_w_cache = {}

def dynamic_batches(ranges, N, min_batch=8):
    """
    Découpe `ranges` en batches dynamiques:
    - s'arrête et démarre un nouveau batch dès qu'un triplet a k == N
    - étend le batch tant que la largeur (k-j) n'excède pas la largeur max actuelle
      ou que le batch n'a pas atteint min_batch
    - autorise un batch plus petit que min_batch uniquement si la coupure est due à k == N

    Retourne une liste de listes :
      [batch_size, current_ranges, max_h, max_w]
    où max_h = max(j - i) et max_w = max(k - j) sur le batch.
    """
    m = len(ranges)
    offset = 0
    batches = []

    while offset < m:
        # init du batch
        i0, j0, k0 = ranges[offset]
        max_w = k0 - j0
        max_h = j0 - i0
        batch_size = 1
        hit_N = False

        # essayer d'étendre le batch
        while offset + batch_size < m:
            i1, j1, k1 = ranges[offset + batch_size]
            w1 = k1 - j1
            h1 = j1 - i1

            # coupure forcée sur k == N
            if k1 == N:
                hit_N = True
                break

            # si ca change, on stoppe
            if w1 != max_w or h1 != max_h:
                break
            batch_size += 1

        # extraction du batch
        current_ranges = ranges[offset:offset + batch_size]
        batches.append([batch_size, current_ranges, max_h, max_w])

        # avancer
        offset += batch_size

    return batches


__workspace_cache = {}  # key: (dtype.str, b, h, w) -> dict(S, U, Z)

def _get_workspace(dtype, b, h, w):
    """
    Réutilise des buffers 3D (b,h,w) pour éviter les allocs répétées.
    S: buffer polyvalent (S12 puis X=R11^{-1}S)
    U: U12 (arrondi) puis delta
    Z: Uf12 (résultat GEMM)
    """
    dtype = cp.dtype(dtype)
    key = (dtype.str, int(b), int(h), int(w))
    ws = __workspace_cache.get(key)
    if ws is None:
        ws = {
            "S": cp.empty((h, w, b), dtype=dtype, order='F'),
            "U": cp.empty((h, w, b), dtype=dtype, order='F'),
            "Z": cp.empty((h, w, b), dtype=dtype, order='F'),
        }
        __workspace_cache[key] = ws
    return ws["S"], ws["U"], ws["Z"]




from cupy_backends.cuda.libs import cublas as _cublas
from cupy.cuda import device as _device

_exec_stream = cp.cuda.Stream(non_blocking=True)

# cache par (n, dtype, ptr_R, ptr_U)
_graphs_cache = {}  # key -> state dict


def _alloc_batch_inputs(dtype, b, h, w):
    """Buffers d'entrée capturés (remplis avant chaque launch)."""
    Rin   = cp.empty((h, h, b), dtype=dtype, order='F')  # R11
    R12in = cp.empty((h, w, b), dtype=dtype, order='F')  # R12
    U22in = cp.empty((w, w, b), dtype=dtype, order='F')  # U22
    UfLin = cp.empty((h, h, b), dtype=dtype, order='F')  # Uf_left
    return Rin, R12in, U22in, UfLin


# État pour le patch cuBLAS
_cublas_handle = None
_orig_get_handle = None
_orig_set_stream = None
_cublas_patched = False

def _enable_cublas_graph_capture():
    """
    Crée un handle cuBLAS lié à _exec_stream et neutralise cublas.setStream()
    pour permettre la capture CUDA Graph, sans changer le stream pendant la capture.
    """
    global _cublas_handle, _orig_get_handle, _orig_set_stream, _cublas_patched
    if _cublas_patched:
        return
    _cublas_handle = _cublas.create()
    _cublas.setStream(_cublas_handle, _exec_stream.ptr)  # bind au stream d'exec

    _orig_get_handle = _device.get_cublas_handle
    _orig_set_stream = _cublas.setStream

    # Cupy demande un handle à chaque op : on renvoie le nôtre (déjà bindé)
    _device.get_cublas_handle = lambda: _cublas_handle
    # Pendant capture, empêcher tout changement de stream
    _cublas.setStream = lambda handle, stream: None

    _cublas_patched = True

def _disable_cublas_graph_capture():
    """Optionnel: restaurer l’état initial si tu veux débrancher le hack."""
    global _cublas_handle, _orig_get_handle, _orig_set_stream, _cublas_patched
    if not _cublas_patched:
        return
    _device.get_cublas_handle = _orig_get_handle
    _cublas.setStream = _orig_set_stream
    try:
        _cublas.destroy(_cublas_handle)
    except Exception:
        pass
    _cublas_handle = None
    _orig_get_handle = None
    _orig_set_stream = None
    _cublas_patched = False
# --- imports & libcublas comme avant ---
import ctypes
import ctypes.util
import cupy as cp
_libcublas = ctypes.cdll.LoadLibrary(ctypes.util.find_library('cublas'))
c_void_p, c_int, c_longlong, c_double = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_double
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_SIDE_LEFT = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_DIAG_NON_UNIT = 0

_libcublas.cublasCreate_v2.argtypes       = [ctypes.POINTER(c_void_p)]
_libcublas.cublasDestroy_v2.argtypes      = [c_void_p]
_libcublas.cublasSetStream_v2.argtypes    = [c_void_p, c_void_p]
_libcublas.cublasDgemmStridedBatched.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_int,
    ctypes.POINTER(c_double),
    c_void_p, c_int, c_longlong,
    c_void_p, c_int, c_longlong,
    ctypes.POINTER(c_double),
    c_void_p, c_int, c_longlong,
    c_int
]
_libcublas.cublasDtrsmBatched.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int,
    c_int, c_int,
    ctypes.POINTER(c_double),
    c_void_p, c_int,   # Aarray (device pointer to device pointers), lda
    c_void_p, c_int,   # Barray (device pointer to device pointers), ldb
    c_int
]

def make_cublas_handle_for_stream(stream: cp.cuda.Stream):
    h = c_void_p()
    err = _libcublas.cublasCreate_v2(ctypes.byref(h))
    if err != 0: raise RuntimeError(f"cublasCreate_v2 failed: {err}")
    _libcublas.cublasSetStream_v2(h, c_void_p(stream.ptr))
    return h

def _assert_f_slices(X: cp.ndarray):
    # vérifie que chaque slice 2D (dernière dim = batch) est F-contigue
    assert X.ndim == 3
    for k in (0, X.shape[2]-1):
        if k >= 0:
            assert X[..., k].flags.f_contiguous, f"slice {k} not F-contiguous"

def _dev_ptr_array_last_axis(base: cp.ndarray) -> cp.ndarray:
    """ Tableau device de pointeurs vers chaque slice 2D base[..., i]. """
    assert base.ndim == 3
    b = base.shape[2]
    step_bytes = base.strides[2]              # << clé : stride du batch
    first = base.data.ptr
    host = (np.uintp(first) + np.arange(b, dtype=np.uintp)*np.uintp(step_bytes))
    return cp.asarray(host)

def trsm_one(handle, A_ptr, B_ptr, h, w, *, upper=True, trans=False, diag_non_unit=True, dtype=cp.float64):
    side = CUBLAS_SIDE_LEFT
    uplo = CUBLAS_FILL_MODE_UPPER if upper else CUBLAS_FILL_MODE_LOWER
    op   = CUBLAS_OP_T if trans else CUBLAS_OP_N
    diag = CUBLAS_DIAG_NON_UNIT if diag_non_unit else CUBLAS_DIAG_UNIT
    lda = h; ldb = h
    if dtype == cp.float64:
        alpha = c_double(1.0)
        err = _libcublas.cublasDtrsm_v2(handle, side, uplo, op, diag,
                                        c_int(h), c_int(w), byref(alpha),
                                        c_void_p(A_ptr), c_int(lda),
                                        c_void_p(B_ptr), c_int(ldb))
    else:
        alpha = c_float(1.0)
        err = _libcublas.cublasStrsm_v2(handle, side, uplo, op, diag,
                                        c_int(h), c_int(w), byref(alpha),
                                        c_void_p(A_ptr), c_int(lda),
                                        c_void_p(B_ptr), c_int(ldb))
    if err != 0:
        raise RuntimeError(f"cublas*trsm_v2 status={err}")

def dgemm_sb_last(h, A, B, C, alpha=1.0, beta=0.0):
    """
    Strided-batched GEMM avec batch en dernière dim :
      A:(m,k,b,F), B:(k,n,b,F), C:(m,n,b,F)
    """
    _assert_f_slices(A); _assert_f_slices(B); _assert_f_slices(C)
    m, k, b = A.shape
    k2, n, b2 = B.shape
    assert k2 == k and b2 == b and C.shape == (m, n, b)
    lda, ldb, ldc = m, k, m
    strideA = A.strides[2] // A.itemsize
    strideB = B.strides[2] // B.itemsize
    strideC = C.strides[2] // C.itemsize
    a_, b_ = c_double(alpha), c_double(beta)
    err = _libcublas.cublasDgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        ctypes.byref(a_),
        c_void_p(A.data.ptr), lda, c_longlong(strideA),
        c_void_p(B.data.ptr), ldb, c_longlong(strideB),
        ctypes.byref(b_),
        c_void_p(C.data.ptr), ldc, c_longlong(strideC),
        c_int(b)
    )
    if err != 0: raise RuntimeError(f"dgemmStridedBatched failed: {err}")

def _chk_last_axis_F(name, X, shape):
    assert X.shape == shape, f"{name}: shape={X.shape}, attendu={shape}"
    # Chaque slice 2D (…,k) doit être F-contigue (col-major)
    assert X[...,0].flags.f_contiguous, f"{name}: slice 0 non F-contigue"
    # stride batch (en bytes) doit être >= m*n*itemsize
    m, n, b = shape
    sb = X.strides[2]
    assert sb >= m*n*X.itemsize, f"{name}: stride batch trop petit ({sb} bytes)"

_h_cublas = None
def _get_cublas_handle():
    global _h_cublas
    if _h_cublas is None:
        _h_cublas = make_cublas_handle_for_stream(_exec_stream)  # << lie le handle au stream capturé
    return _h_cublas
def _build_state_and_capture(n, dtype, R_gpu, U_gpu, Uf_gpu, ranges_np_full, batches):
    """
    Prépare I/J, workspaces et buffers d'entrée (layout (h,*,b), F-order).
    Fait un warmup (= même séquence que la capture) puis capture 1 graphe par batch.
    """
    key = (int(n), cp.dtype(dtype).str, int(R_gpu.data.ptr), int(U_gpu.data.ptr))
    if key in _graphs_cache:
        return _graphs_cache[key]

    state = {
        "I_cache": {},
        "J_cache": {},
        "workspaces": {},   # idx -> (S, U, Z)   (h,w,b) F-order
        "g_inputs": {},     # idx -> (Rin, R12in, U22in, UfLin)
        "graphs": {},       # idx -> cupy.cuda.graph.Graph
        "batches": batches,
        "ranges_np_full": ranges_np_full,
        "Uf_gpu": Uf_gpu,   # float working copy (pointeur stable)
    }

    offset = 0
    for idx, (batch_size, _ranges, max_h, max_w) in enumerate(batches):
        b, h, w = int(batch_size), int(max_h), int(max_w)

        # i_arr, j_arr pour ce batch
        ranges_np = ranges_np_full[offset: offset + batch_size]
        i_arr, j_arr = ranges_np[:, 0], ranges_np[:, 1]

        # indices (hors capture)
        rows_h = _rows_h_cache.setdefault(h, cp.arange(h, dtype=cp.int32))
        cols_w = _cols_w_cache.setdefault(w, cp.arange(w, dtype=cp.int32))
        I = i_arr[:, None] + rows_h[None, :]  # (b,h)
        J = j_arr[:, None] + cols_w[None, :]  # (b,w)
        state["I_cache"][idx] = I
        state["J_cache"][idx] = J

        # workspaces réutilisés (h,w,b)
        S, U, Z = _get_workspace(dtype, b, h, w)
        state["workspaces"][idx] = (S, U, Z)

        # buffers d'entrée (h,*,b)
        Rin, R12in, U22in, UfLin = _alloc_batch_inputs(dtype, b, h, w)
        _chk_last_axis_F("Rin", Rin, (h,h,b)); _chk_last_axis_F("R12in", R12in, (h,w,b))
        _chk_last_axis_F("U22in", U22in, (w,w,b)); _chk_last_axis_F("UfLin", UfLin, (h,h,b))
        _chk_last_axis_F("S", S, (h,w,b)); _chk_last_axis_F("U", U, (h,w,b)); _chk_last_axis_F("Z", Z, (h,w,b))
        state["g_inputs"][idx] = (Rin, R12in, U22in, UfLin)

        offset += batch_size

    h_cublas = _get_cublas_handle()

    # ---------- 2) WARMUP : même séquence que la capture ----------
    offset = 0
    with _exec_stream:
        for idx, (bs, _rng, h, w) in enumerate(batches):
            b, h, w = int(bs), int(h), int(w)
            I = state["I_cache"][idx]; J = state["J_cache"][idx]
            S, U, Z = state["workspaces"][idx]
            Rin, R12in, U22in, UfLin = state["g_inputs"][idx]

            # Gather (hors capture)
            R11 = R_gpu[I[:,:,None], I[:,None,:]]             # (b,h,h)
            R12 = R_gpu[I[:,:,None], J[:,None,:]]             # (b,h,w)
            U22 = state["Uf_gpu"][J[:,:,None], J[:,None,:]]   # (b,w,w)
            UfL = state["Uf_gpu"][I[:,:,None], I[:,None,:]]   # (b,h,h)
            Rin[...]   = cp.asfortranarray(cp.transpose(R11, (1,2,0)))
            R12in[...] = cp.asfortranarray(cp.transpose(R12, (1,2,0)))
            U22in[...] = cp.asfortranarray(cp.transpose(U22, (1,2,0)))
            UfLin[...] = cp.asfortranarray(cp.transpose(UfL,  (1,2,0)))

            # --- cœur math : identique à la capture ---
            # S_R = R12·U22
            dgemm_sb_last(h_cublas, R12in, U22in, S, 1.0, 0.0)
            # U12' = round( - R11^{-1} S_R )
            U[...] = S
            for k in range(b):
                trsm_one(h_cublas, Rin[...,k].data.ptr, U[...,k].data.ptr,
                         h, w, upper=True, trans=False, diag_non_unit=True, dtype=dtype)
            cp.multiply(U, -1, out=U); cp.rint(U, out=U)
            # Z = R11·U12' ; S_R += Z
            dgemm_sb_last(h_cublas, Rin, U, Z, 1.0, 0.0)
            cp.add(S, Z, out=S)
            # write R block maintenant (avant d'écraser S)
            S_bhw = cp.transpose(S, (2,0,1))
            R_gpu[I[:, :, None], J[:, None, :]] = S_bhw

            # S_U = Uf_left · U12' (réutilise S)
            dgemm_sb_last(h_cublas, UfLin, U, S, 1.0, 0.0)
            S_bhw = cp.transpose(S, (2,0,1))
            state["Uf_gpu"][I[:, :, None], J[:, None, :]] = S_bhw
    _exec_stream.synchronize()

    # ---------- 3) CAPTURE : 1 graphe par batch ----------
    for idx, (bs, _rng, h, w) in enumerate(batches):
        b, h, w = int(bs), int(h), int(w)
        I = state["I_cache"][idx]; J = state["J_cache"][idx]
        S, U, Z = state["workspaces"][idx]
        Rin, R12in, U22in, UfLin = state["g_inputs"][idx]

        with _exec_stream:
            _exec_stream.begin_capture()
            # S_R = R12·U22
            dgemm_sb_last(h_cublas, R12in, U22in, S, 1.0, 0.0)
            # U12' = round( - R11^{-1} S_R )
            U[...] = S
            for k in range(b):
                trsm_one(h_cublas, Rin[...,k].data.ptr, U[...,k].data.ptr,
                         h, w, upper=True, trans=False, diag_non_unit=True, dtype=dtype)
            cp.multiply(U, -1, out=U); cp.rint(U, out=U)
            # Z = R11·U12' ; S_R += Z ; écrire R
            dgemm_sb_last(h_cublas, Rin, U, Z, 1.0, 0.0)
            cp.add(S, Z, out=S)
            S_bhw = cp.transpose(S, (2,0,1))
            R_gpu[I[:, :, None], J[:, None, :]] = S_bhw
            # S_U = Uf_left · U12' ; écrire Uf
            dgemm_sb_last(h_cublas, UfLin, U, S, 1.0, 0.0)
            S_bhw = cp.transpose(S, (2,0,1))
            state["Uf_gpu"][I[:, :, None], J[:, None, :]] = S_bhw
            g = _exec_stream.end_capture()
        # garder le graph et pré‐uploader
        state["graphs"][idx] = g
        try:
            g.upload(_exec_stream)
        except Exception:
            pass

    _graphs_cache[key] = state
    return state


def seysen_reduce_gpu(R_gpu, U_gpu):
    """
    Version capturable math‐correcte. N'écrit U_gpu (int) qu'à la fin.
    """
    n = R_gpu.shape[0]
    dtype = R_gpu.dtype
    min_batch, area_batch = 8, 32*32

    # Ranges/coupes
    if n not in __reduction_cache:
        base_cases, ranges = __reduction_ranges(n)
        i_arr = cp.array(base_cases, dtype=cp.int32)
        batches = dynamic_batches(ranges, n, min_batch=min_batch)
        ranges_np_full = cp.asarray(ranges, dtype=cp.int32)
        __reduction_cache[n] = (base_cases, ranges, i_arr, ranges_np_full, batches)
    else:
        base_cases, ranges, i_arr, ranges_np_full, batches = __reduction_cache[n]

    # Base cases – NE PAS écraser v (vue sur R)
    if base_cases:
        u = R_gpu[i_arr, i_arr]
        v = R_gpu[i_arr, i_arr + 1]
        t = -cp.rint(v / u)                              # t calculé à part
        U_gpu[i_arr, i_arr + 1] = t.astype(U_gpu.dtype)  # écrit U (int)
        R_gpu[i_arr, i_arr + 1] += u * t                 # met à jour R

    # Copie float de U (stable pour graphs)
    key = (int(n), cp.dtype(dtype).str, int(R_gpu.data.ptr), int(U_gpu.data.ptr))
    if key in _graphs_cache:
        state = _graphs_cache[key]
        Uf_gpu = state["Uf_gpu"]
        Uf_gpu[...] = U_gpu  # sync contenu (pas de réallocation)
    else:
        Uf_gpu = U_gpu.astype(dtype, copy=True)
        state = _build_state_and_capture(n, dtype, R_gpu, U_gpu, Uf_gpu,
                                         __reduction_cache[n][3], __reduction_cache[n][4])

    # Boucle principale : grands batches → graph.launch, petits → direct
    offset = 0
    with _exec_stream:
        for idx, (batch_size, current_ranges, max_h, max_w) in enumerate(state["batches"]):
            b, h, w = int(batch_size), int(max_h), int(max_w)
            big = (b > 1) and not (b < min_batch and h*w < area_batch)
            if big:
                I = state["I_cache"][idx]; J = state["J_cache"][idx]
                Rin, R12in, U22in, UfLin = state["g_inputs"][idx]
                # Gather (hors capture, mais juste avant launch)
                R11 = R_gpu[I[:,:,None], I[:,None,:]]             # (b,h,h)
                R12 = R_gpu[I[:,:,None], J[:,None,:]]             # (b,h,w)
                U22 = Uf_gpu[J[:,:,None], J[:,None,:]]            # (b,w,w)
                UfL = Uf_gpu[I[:,:,None], I[:,None,:]]            # (b,h,h)
                Rin[...]   = cp.asfortranarray(cp.transpose(R11, (1,2,0)))
                R12in[...] = cp.asfortranarray(cp.transpose(R12, (1,2,0)))
                U22in[...] = cp.asfortranarray(cp.transpose(U22, (1,2,0)))
                UfLin[...] = cp.asfortranarray(cp.transpose(UfL,  (1,2,0)))
                state["graphs"][idx].launch(stream=_exec_stream)
            else:
                # Chemin direct (petits blocs) – identique math
                ranges_np = state["ranges_np_full"][offset: offset + batch_size]
                for (i, j, k) in ranges_np.tolist():
                    S12p = R_gpu[i:j, j:k].dot(Uf_gpu[j:k, j:k])   # S_R
                    S11  = R_gpu[i:j, i:j]
                    U12p = S12p.copy()
                    solve_triangular(S11, U12p, lower=False, overwrite_b=True)
                    cp.multiply(U12p, -1, out=U12p); cp.rint(U12p, out=U12p)
                    Z = cp.empty_like(S12p)
                    cp.matmul(S11, U12p, out=Z)
                    cp.add(S12p, Z, out=S12p)
                    R_gpu[i:j, j:k] = S12p                          # write R
                    Uf_blk = Uf_gpu[i:j, i:j]
                    U12 = cp.empty_like(U12p)
                    cp.matmul(Uf_blk, U12p, out=U12)
                    Uf_gpu[i:j, j:k] = U12                          # update Uf
            offset += batch_size

    # On n’écrit U_gpu (int) qu’à la fin : arrondi, pas cast
    U_gpu[:, :] = cp.rint(Uf_gpu).astype(U_gpu.dtype)