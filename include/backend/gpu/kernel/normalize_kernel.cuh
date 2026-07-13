#pragma once
#include <stdint.h>

// base-2^kappa normalizasyonu: her katsayı pozisyonu bağımsız işlenir.
// Carry az-anlamlı limb'den (l-1) çok-anlamlı limb'e (0) doğru akar.
// res[i*res_sl+k] ∈ [-(2^(kappa-1)), 2^(kappa-1))  for all i, k
//
// res_sl/a_sl are the element stride between consecutive limbs (== n for a
// contiguous buffer; may be larger when normalizing a strided view, e.g.
// one component of an interleaved GLWE ciphertext).
//
// res and a may have DIFFERENT limb counts (res_size vs a_size), mirroring
// spqlios' vec_znx_normalize_base2k_ref: carry propagates through any of a's
// limbs beyond res_size (computed but discarded), and any of res's limbs
// beyond a_size are zeroed. Needed wherever normalize_glwe's operands have
// different params (e.g. an automorphism KSK's own precision vs. the GLWE
// ciphertext being transformed, as in glwe_trace_expand).
//
// The plain, single-limb-count normalize case (res_size == a_size,
// res_sl == a_sl) is just this same kernel with both loops that handle the
// size mismatch trivially skipped (zero iterations) — see gpu_normalize_base2k
// in vec_znx_arith_host.cu — so there is no separate non-sized kernel.
__global__ void normalize_base2k_sized_kernel(int64_t* res, int res_size, int64_t res_sl, const int64_t* a, int a_size,
                                              int64_t a_sl, int n, int kappa);
