#ifndef POLY_ADD_HOST_H
#define POLY_ADD_HOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU coefficient-wise polynomial addition modulo q in ZnX.
 * c[i] = (a[i] + b[i]) mod q for all i in [0, n). Host memory in/out.
 * @return 0 on success, -1 on CUDA error
 */
int gpu_poly_add_mod(const uint64_t* a, const uint64_t* b, uint64_t* c, uint64_t n, uint64_t q);

/**
 * Same as gpu_poly_add_mod but also reports kernel-only execution time in milliseconds
 * (excludes host-to-device and device-to-host memory transfers).
 * @return 0 on success, -1 on CUDA error
 */
int gpu_poly_add_mod_timed(const uint64_t* a, const uint64_t* b, uint64_t* c, uint64_t n, uint64_t q,
                           float* gpu_kernel_ms);

#ifdef __cplusplus
}
#endif

#endif  // POLY_ADD_HOST_H
