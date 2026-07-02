#include <cmath>
#include <cstdlib>

#include "core/ggsw/ggsw_ciphertext.h"
#include "core/ggsw/ggsw_params.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_params.h"
#include "gpu/common/gpu_common.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/kernel/ntt_poly_mult_kernel.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

extern "C" {

int pvda_is_device_pointer(const void* ptr)
{
    return is_gpu_device_pointer(ptr) ? 1 : 0;
}

int64_t* pvda_gpu_upload(const int64_t* host_ptr, size_t n_elements)
{
    Data64s* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, n_elements * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr, n_elements * sizeof(int64_t), cudaMemcpyHostToDevice));
    return (int64_t*)d_ptr;
}

void pvda_gpu_download(int64_t* host_ptr, const int64_t* device_ptr, size_t n_elements)
{
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, n_elements * sizeof(int64_t), cudaMemcpyDeviceToHost));
}

int64_t* pvda_gpu_alloc(size_t n_elements)
{
    Data64s* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, n_elements * sizeof(int64_t)));
    return (int64_t*)d_ptr;
}

void pvda_gpu_free(int64_t* device_ptr) { cudaFree(device_ptr); }

int64_t* pvda_glwe_to_device(const GLWECiphertext* glwe)
{
    size_t n_elems = (size_t)glwe->params->ciphertext_nb_limbs * (size_t)glwe->params->nn;
    return pvda_gpu_upload((const int64_t*)glwe->vec, n_elems);
}

void pvda_glwe_from_device(GLWECiphertext* result, const int64_t* d_data)
{
    size_t n_elems = (size_t)result->params->ciphertext_nb_limbs * (size_t)result->params->nn;
    pvda_gpu_download((int64_t*)result->vec, d_data, n_elems);
}

int64_t* pvda_ggsw_to_device(const GGSWCiphertext* ggsw)
{
    // nrows * ncols * nn  =  ciphertext_nb_limbs_tilde * params_glwe->ciphertext_nb_limbs * params_glwe->nn
    size_t n_elems = (size_t)ggsw->params->ciphertext_nb_limbs_tilde *
                     (size_t)ggsw->params->params_glwe->ciphertext_nb_limbs *
                     (size_t)ggsw->params->params_glwe->nn;
    return pvda_gpu_upload((const int64_t*)ggsw->mat, n_elems);
}

void gpu_ggsw_external_product_device(const int64_t* d_glwe, const int64_t* d_ggsw, int64_t* d_result, size_t n,
                                      size_t nrows, size_t ncols)
{
    NTTParameterGenerator& gen = NTTParameterGenerator::instance();
    gen.initialize(n);

    Modulus64  mod      = gen.get_modulus();
    Root64*    ntt_tab  = gen.get_ntt_table(n);
    Root64*    intt_tab = gen.get_intt_table(n);
    Ninverse64 n_inv    = gen.get_n_inv(n);

    int n_power = (int)std::log2((double)n);

    gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
                                                 .ntt_type       = gpuntt::FORWARD,
                                                 .ntt_layout     = gpuntt::PerPolynomial,
                                                 .reduction_poly = gpuntt::X_N_plus,
                                                 .zero_padding   = false,
                                                 .mod_inverse    = 0,
                                                 .stream         = 0};

    gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
                                                 .ntt_type       = gpuntt::INVERSE,
                                                 .ntt_layout     = gpuntt::PerPolynomial,
                                                 .reduction_poly = gpuntt::X_N_plus,
                                                 .zero_padding   = false,
                                                 .mod_inverse    = n_inv,
                                                 .stream         = 0};

    // NTT(GLWE): nrows polynomials — input already on device, no H2D copy
    VEC_GPU<Data64> d_a_ntt(n * nrows);
    gpuntt::GPU_NTT((Data64s*)d_glwe, d_a_ntt.data(), ntt_tab, mod, cfg_fwd, (int)nrows);

    // NTT(GGSW): nrows*ncols polynomials — input already on device
    VEC_GPU<Data64> d_m_ntt(n * nrows * ncols);
    gpuntt::GPU_NTT((Data64s*)d_ggsw, d_m_ntt.data(), ntt_tab, mod, cfg_fwd, (int)(nrows * ncols));

    // VMP accumulate: d_c_ntt[j*n+k] = sum_i d_a_ntt[i*n+k] * d_m_ntt[(i*ncols+j)*n+k]
    VEC_GPU<Data64> d_c_ntt(n * ncols);
    int threads = 256;
    int blocks  = ((int)(n * ncols) + threads - 1) / threads;
    vmp_accumulate_kernel<<<blocks, threads>>>(d_c_ntt.data(), d_a_ntt.data(), d_m_ntt.data(), mod, (int)n, (int)nrows,
                                              (int)ncols);
    CUDA_CHECK(cudaGetLastError());

    // INTT: write directly into d_result (already on device, no D2H copy)
    gpuntt::GPU_INTT(d_c_ntt.data(), (Data64s*)d_result, intt_tab, mod, cfg_inv, (int)ncols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // extern "C"
