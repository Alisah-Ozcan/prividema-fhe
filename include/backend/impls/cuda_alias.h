#ifndef CUDA_ALIAS_H
#define CUDA_ALIAS_H

#include "backend.h"
#include "backend_private.h"

/**
 * @file cuda_alias.h
 *
 * Virtual table of the CUDA backend. Each operation runs on the GPU when its operands are in
 * GPU memory and on the embedded spqlios backend (the backend's host_vt) when they are in host
 * memory; operands split between the two are rejected with an error. Buffers allocated through
 * the table (pvda_new_vec_znx_dft, ...) are host buffers: GPU objects are created with the
 * backend-placed constructors (new_glwe_on, ...).
 */

/**
 * @brief Fills vt with the CUDA operations. host_vt must already hold the spqlios table.
 */
void pvda_fill_cuda(struct pvda_virtual_table* vt);

#endif
