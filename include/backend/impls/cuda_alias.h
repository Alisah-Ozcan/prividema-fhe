#ifndef CUDA_ALIAS_H
#define CUDA_ALIAS_H

#include "backend.h"
#include "backend_private.h"

/**
 * @file cuda_alias.h
 *
 * Virtual table of the CUDA backend. The opaque backend objects (DFT vectors, prepared
 * polynomials, vectors and matrices) are allocated in GPU memory. Every operation with a GPU
 * implementation runs on the GPU: the coefficient-domain vectors (ZnxVec), which stay in host
 * memory, are uploaded for it and a host result is downloaded. znx_small_product (no GPU
 * implementation), the RNG and the number-theory entries run on the embedded spqlios backend (the
 * backend's host_vt).
 */

/**
 * @brief Creates a CUDA backend (what pvda_new_backend returns in a build with ENABLE_CUDA=ON)
 *
 * @return NULL if no CUDA device is available
 */
PvdaBackend* pvda_new_cuda_backend(int nn);

/**
 * @brief Fills vt with the CUDA operations. host_vt must already hold the spqlios table.
 */
void pvda_fill_cuda(struct pvda_virtual_table* vt);

#endif
