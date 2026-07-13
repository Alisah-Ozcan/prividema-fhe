#include <cmath>
#include <cstdlib>

#include "core/ggsw/ggsw_ciphertext.h"
#include "core/ggsw/ggsw_params.h"
#include "core/ggsw/glwegadget_ciphertext.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_params.h"
#include "gpu/common/gpu_common.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/kernel/ntt_poly_mult_kernel.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

extern "C" {

int pvda_is_device_pointer(const void* ptr) { return is_gpu_device_pointer(ptr) ? 1 : 0; }

int64_t* pvda_gpu_upload(const int64_t* host_ptr, size_t n_elements)
{
	Data64s* d_ptr = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_ptr, n_elements * sizeof(int64_t)));
	CUDA_CHECK(
	    cudaMemcpyAsync(d_ptr, host_ptr, n_elements * sizeof(int64_t), cudaMemcpyHostToDevice, gpu_active_stream));
	return (int64_t*)d_ptr;
}

void pvda_gpu_download(int64_t* host_ptr, const int64_t* device_ptr, size_t n_elements)
{
	// Downloads cross back into host memory that callers read immediately on
	// return (no further stream sync at the call site), so this primitive
	// keeps its synchronous contract despite using the active stream.
	CUDA_CHECK(
	    cudaMemcpyAsync(host_ptr, device_ptr, n_elements * sizeof(int64_t), cudaMemcpyDeviceToHost, gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

int64_t* pvda_gpu_alloc(size_t n_elements)
{
	Data64s* d_ptr = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_ptr, n_elements * sizeof(int64_t)));
	return (int64_t*)d_ptr;
}

void pvda_gpu_free(int64_t* device_ptr) { cudaFree(device_ptr); }

void pvda_gpu_copy(int64_t* dst, const int64_t* src, size_t n_elements)
{
	CUDA_CHECK(cudaMemcpyAsync(dst, src, n_elements * sizeof(int64_t), cudaMemcpyDeviceToDevice, gpu_active_stream));
}

void pvda_gpu_zero(int64_t* dst, size_t n_elements)
{
	CUDA_CHECK(cudaMemsetAsync(dst, 0, n_elements * sizeof(int64_t), gpu_active_stream));
}

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
	                 (size_t)ggsw->params->params_glwe->ciphertext_nb_limbs * (size_t)ggsw->params->params_glwe->nn;
	return pvda_gpu_upload((const int64_t*)ggsw->mat, n_elems);
}

GLWECiphertext* pvda_new_glwe_device(const GLWEParams* params)
{
	GLWECiphertext* glwe = (GLWECiphertext*)malloc(sizeof(GLWECiphertext));
	if (!glwe) return nullptr;

	// ciphertext_nb_limbs * nn == glwe_coef_number(params) — computed directly (rather
	// than calling glwe_coef_number) since glwe_params.h's declarations aren't wrapped in
	// extern "C", so calling it from this TU (compiled as C++) would look for a mangled
	// symbol that the C-compiled definition never exports.
	size_t n_elems = (size_t)params->ciphertext_nb_limbs * (size_t)params->nn;

	glwe->params = params;
	glwe->vec    = (VecBiv*)pvda_gpu_alloc(n_elems);
	return glwe;
}

GLWEGadgetCiphertext* pvda_new_glwegadget_device(const GLWEGadgetParams* params)
{
	GLWEGadgetCiphertext* glwegad = (GLWEGadgetCiphertext*)malloc(sizeof(GLWEGadgetCiphertext));
	if (!glwegad) return nullptr;

	// l_tilde * ciphertext_nb_limbs * nn == glwegadget_coef_number(params) — computed
	// directly rather than calling glwegadget_coef_number, same extern "C" mangling
	// reason as pvda_new_glwe_device above.
	size_t n_elems =
	    (size_t)params->l_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs * (size_t)params->params_glwe->nn;

	glwegad->params = params;
	glwegad->mat    = (MatBiv*)pvda_gpu_alloc(n_elems);
	pvda_gpu_zero((int64_t*)glwegad->mat, n_elems);
	return glwegad;
}

GGSWCiphertext* pvda_new_ggsw_device(const GGSWParams* params)
{
	GGSWCiphertext* ggsw = (GGSWCiphertext*)malloc(sizeof(GGSWCiphertext));
	if (!ggsw) return nullptr;

	// ciphertext_nb_limbs_tilde * ciphertext_nb_limbs * nn == ggsw_coef_number(params) —
	// computed directly, same extern "C" mangling reason as pvda_new_glwe_device above.
	size_t n_elems = (size_t)params->ciphertext_nb_limbs_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs *
	                 (size_t)params->params_glwe->nn;

	ggsw->params = params;
	ggsw->mat    = (MatBiv*)pvda_gpu_alloc(n_elems);
	pvda_gpu_zero((int64_t*)ggsw->mat, n_elems);
	return ggsw;
}

GLWEGadgetCiphertextPrep* pvda_new_glwegadget_prep_device(const GLWEGadgetParams* params)
{
	GLWEGadgetCiphertextPrep* prep = (GLWEGadgetCiphertextPrep*)malloc(sizeof(GLWEGadgetCiphertextPrep));
	if (!prep) return nullptr;

	// Placeholder footprint — doesn't need to match the true DFT-domain
	// size, since glwegadget_prepare_gpu always allocates a fresh buffer
	// and replaces ->mat itself. This one only needs to exist long enough
	// for pvda_is_device_pointer to see it (callers free it themselves
	// right before the real prepare call, see prepare_automorphism_key).
	size_t n_elems =
	    (size_t)params->l_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs * (size_t)params->params_glwe->nn;

	prep->params = params;
	prep->mat    = (MatBivDFT*)pvda_gpu_alloc(n_elems);
	return prep;
}

GGSWCiphertextPrep* pvda_new_ggsw_prep_device(const GGSWParams* params)
{
	GGSWCiphertextPrep* prep = (GGSWCiphertextPrep*)malloc(sizeof(GGSWCiphertextPrep));
	if (!prep) return nullptr;

	size_t n_elems = (size_t)params->ciphertext_nb_limbs_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs *
	                 (size_t)params->params_glwe->nn;

	prep->params = params;
	prep->mat    = (MatBivDFT*)pvda_gpu_alloc(n_elems);
	return prep;
}

void gpu_ggsw_external_product_device(const int64_t* d_glwe, const int64_t* d_ggsw, int64_t* d_result, size_t n,
                                      size_t nrows, size_t ncols, size_t a_limbs)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod    = gen.get_modulus();
	Root64* ntt_tab  = gen.get_ntt_table(n);
	Root64* intt_tab = gen.get_intt_table(n);
	Ninverse64 n_inv = gen.get_n_inv(n);

	int n_power = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::INVERSE,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = n_inv,
	                                             .stream         = gpu_active_stream};

	// NTT(GLWE): nrows polynomials expected by the VMP, but d_glwe only holds `a_limbs`
	// valid polynomials (the GLWE's own limb count, which may differ from both nrows and
	// ncols — e.g. when the GGSW selector's associated GLWE params differ from the
	// ciphertext being multiplied). Zero-pad the missing rows instead of reading past the
	// buffer (mirrors spqlios' implicit zero-padding on the CPU path).
	VEC_GPU<Data64> d_a_ntt(n * nrows);
	size_t valid_rows = nrows < a_limbs ? nrows : a_limbs;
	if (valid_rows < nrows)
		CUDA_CHECK(cudaMemsetAsync(d_a_ntt.data() + valid_rows * n, 0, (nrows - valid_rows) * n * sizeof(Data64),
		                           gpu_active_stream));
	gpuntt::GPU_NTT((Data64s*)d_glwe, d_a_ntt.data(), ntt_tab, mod, cfg_fwd, (int)valid_rows);

	// NTT(GGSW): nrows*ncols polynomials — input already on device
	VEC_GPU<Data64> d_m_ntt(n * nrows * ncols);
	gpuntt::GPU_NTT((Data64s*)d_ggsw, d_m_ntt.data(), ntt_tab, mod, cfg_fwd, (int)(nrows * ncols));

	// VMP accumulate: d_c_ntt[j*n+k] = sum_i d_a_ntt[i*n+k] * d_m_ntt[(i*ncols+j)*n+k]
	VEC_GPU<Data64> d_c_ntt(n * ncols);
	int threads = 256;
	int blocks  = ((int)(n * ncols) + threads - 1) / threads;
	vmp_accumulate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_c_ntt.data(), d_a_ntt.data(), d_m_ntt.data(),
	                                                                 mod, (int)n, (int)nrows, (int)ncols);
	CUDA_CHECK(cudaGetLastError());

	// INTT: write directly into d_result (already on device, no D2H copy)
	gpuntt::GPU_INTT(d_c_ntt.data(), (Data64s*)d_result, intt_tab, mod, cfg_inv, (int)ncols);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void ggsw_prepare_gpu(GGSWCiphertextPrep* gpu_prep, const GGSWCiphertext* ggsw)
{
	size_t n     = (size_t)ggsw->params->params_glwe->nn;
	size_t nrows = (size_t)ggsw->params->ciphertext_nb_limbs_tilde;
	size_t ncols = (size_t)ggsw->params->params_glwe->ciphertext_nb_limbs;

	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod   = gen.get_modulus();
	Root64* ntt_tab = gen.get_ntt_table(n);
	int n_power     = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	// Accept both host and device GGSW input
	const int64_t* d_src;
	int64_t* d_tmp = nullptr;
	if (is_gpu_device_pointer(ggsw->mat))
	{
		d_src = (const int64_t*)ggsw->mat;
	}
	else
	{
		d_tmp = pvda_ggsw_to_device(ggsw);
		d_src = d_tmp;
	}

	// GPU_NTT output must be Data64* (unsigned); allocate separately then store as MatBivDFT*
	Data64* d_ntt = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_ntt, nrows * ncols * n * sizeof(Data64)));
	gpuntt::GPU_NTT((Data64s*)d_src, d_ntt, ntt_tab, mod, cfg_fwd, (int)(nrows * ncols));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));

	if (d_tmp) pvda_gpu_free(d_tmp);

	gpu_prep->mat = (MatBivDFT*)d_ntt;
}

void gpu_ggsw_ext_prod_ntt_device(const int64_t* d_glwe, const int64_t* d_ggsw_ntt, int64_t* d_result_ntt, size_t n,
                                  size_t nrows, size_t ncols, size_t a_limbs)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod   = gen.get_modulus();
	Root64* ntt_tab = gen.get_ntt_table(n);
	int n_power     = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	// NTT(GLWE): nrows polynomials expected by the VMP, but d_glwe only holds `a_limbs`
	// valid polynomials (the GLWE's own limb count). Zero-pad the missing rows instead
	// of reading past the buffer when nrows > a_limbs — see gpu_ggsw_external_product_device.
	VEC_GPU<Data64> d_a_ntt(n * nrows);
	size_t valid_rows = nrows < a_limbs ? nrows : a_limbs;
	if (valid_rows < nrows)
		CUDA_CHECK(cudaMemsetAsync(d_a_ntt.data() + valid_rows * n, 0, (nrows - valid_rows) * n * sizeof(Data64),
		                           gpu_active_stream));
	gpuntt::GPU_NTT((Data64s*)d_glwe, d_a_ntt.data(), ntt_tab, mod, cfg_fwd, (int)valid_rows);

	// d_ggsw_ntt is already in NTT domain — use directly
	VEC_GPU<Data64> d_c_ntt(n * ncols);
	int threads = 256;
	int blocks  = ((int)(n * ncols) + threads - 1) / threads;
	vmp_accumulate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    d_c_ntt.data(), d_a_ntt.data(), (const Data64*)d_ggsw_ntt, mod, (int)n, (int)nrows, (int)ncols);
	CUDA_CHECK(cudaGetLastError());

	// Store NTT-domain result — no INTT
	CUDA_CHECK(cudaMemcpyAsync(d_result_ntt, d_c_ntt.data(), n * ncols * sizeof(int64_t), cudaMemcpyDeviceToDevice,
	                           gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void glwe_dft_to_coef_gpu(GLWECiphertext* d_result, const GLWECiphertextDFT* d_glwe_ntt)
{
	size_t n     = (size_t)d_glwe_ntt->params->nn;
	size_t ncols = (size_t)d_glwe_ntt->params->ciphertext_nb_limbs;

	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod    = gen.get_modulus();
	Root64* intt_tab = gen.get_intt_table(n);
	Ninverse64 n_inv = gen.get_n_inv(n);
	int n_power      = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::INVERSE,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = n_inv,
	                                             .stream         = gpu_active_stream};

	// GPU_INTT input is Data64* (unsigned NTT domain); output is Data64s* (signed coef domain)
	gpuntt::GPU_INTT((Data64*)d_glwe_ntt->vec, (Data64s*)d_result->vec, intt_tab, mod, cfg_inv, (int)ncols);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

int64_t* pvda_glwegadget_to_device(const GLWEGadgetCiphertext* glwegad)
{
	size_t n_elems = (size_t)glwegad->params->l_tilde * (size_t)glwegad->params->params_glwe->ciphertext_nb_limbs *
	                 (size_t)glwegad->params->params_glwe->nn;
	return pvda_gpu_upload((const int64_t*)glwegad->mat, n_elems);
}

void glwegadget_prepare_gpu(GLWEGadgetCiphertextPrep* gpu_prep, const GLWEGadgetCiphertext* glwegad)
{
	size_t n     = (size_t)glwegad->params->params_glwe->nn;
	size_t nrows = (size_t)glwegad->params->l_tilde;
	size_t ncols = (size_t)glwegad->params->params_glwe->ciphertext_nb_limbs;

	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod   = gen.get_modulus();
	Root64* ntt_tab = gen.get_ntt_table(n);
	int n_power     = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	const int64_t* d_src;
	int64_t* d_tmp = nullptr;
	if (is_gpu_device_pointer(glwegad->mat))
	{
		d_src = (const int64_t*)glwegad->mat;
	}
	else
	{
		d_tmp = pvda_glwegadget_to_device(glwegad);
		d_src = d_tmp;
	}

	Data64* d_ntt = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_ntt, nrows * ncols * n * sizeof(Data64)));
	gpuntt::GPU_NTT((Data64s*)d_src, d_ntt, ntt_tab, mod, cfg_fwd, (int)(nrows * ncols));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));

	if (d_tmp) pvda_gpu_free(d_tmp);

	gpu_prep->mat = (MatBivDFT*)d_ntt;
}

void gpu_glwegadget_half_prod_device(const int64_t* d_a, const int64_t* d_glwegad_ntt, int64_t* d_result, size_t n,
                                     size_t nrows, size_t ncols_in, size_t ncols_out)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod    = gen.get_modulus();
	Root64* ntt_tab  = gen.get_ntt_table(n);
	Root64* intt_tab = gen.get_intt_table(n);
	Ninverse64 n_inv = gen.get_n_inv(n);
	int n_power      = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::INVERSE,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = n_inv,
	                                             .stream         = gpu_active_stream};

	// NTT(a): nrows=l_tilde polynomials
	VEC_GPU<Data64> d_a_ntt(n * nrows);
	gpuntt::GPU_NTT((Data64s*)d_a, d_a_ntt.data(), ntt_tab, mod, cfg_fwd, (int)nrows);

	// VMP: d_glwegad_ntt is already in NTT domain (from glwegadget_prepare_gpu).
	// The matrix genuinely has ncols_in columns — this must use ncols_in, not
	// the caller's result size, regardless of how the two compare.
	VEC_GPU<Data64> d_c_ntt(n * ncols_in);
	int threads = 256;
	int blocks  = ((int)(n * ncols_in) + threads - 1) / threads;
	vmp_accumulate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    d_c_ntt.data(), d_a_ntt.data(), (const Data64*)d_glwegad_ntt, mod, (int)n, (int)nrows, (int)ncols_in);
	CUDA_CHECK(cudaGetLastError());

	// INTT: only min(ncols_in, ncols_out) columns are real VMP output — never
	// write more than ncols_out columns into d_result (that's the caller's
	// actual buffer capacity), and zero-pad any tail beyond ncols_in if
	// ncols_out is larger.
	size_t common = ncols_in < ncols_out ? ncols_in : ncols_out;
	gpuntt::GPU_INTT(d_c_ntt.data(), (Data64s*)d_result, intt_tab, mod, cfg_inv, (int)common);
	if (ncols_out > common)
		CUDA_CHECK(
		    cudaMemsetAsync(d_result + common * n, 0, (ncols_out - common) * n * sizeof(int64_t), gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_glwegadget_half_prod_ntt_device(const int64_t* d_a, const int64_t* d_glwegad_ntt, int64_t* d_result_ntt,
                                         size_t n, size_t nrows, size_t ncols)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod   = gen.get_modulus();
	Root64* ntt_tab = gen.get_ntt_table(n);
	int n_power     = (int)std::log2((double)n);

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	// NTT(a): nrows=l_tilde polynomials
	VEC_GPU<Data64> d_a_ntt(n * nrows);
	gpuntt::GPU_NTT((Data64s*)d_a, d_a_ntt.data(), ntt_tab, mod, cfg_fwd, (int)nrows);

	// VMP: d_glwegad_ntt is already in NTT domain — result stays in NTT domain (no INTT)
	VEC_GPU<Data64> d_c_ntt(n * ncols);
	int threads = 256;
	int blocks  = ((int)(n * ncols) + threads - 1) / threads;
	vmp_accumulate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    d_c_ntt.data(), d_a_ntt.data(), (const Data64*)d_glwegad_ntt, mod, (int)n, (int)nrows, (int)ncols);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaMemcpyAsync(d_result_ntt, d_c_ntt.data(), n * ncols * sizeof(int64_t), cudaMemcpyDeviceToDevice,
	                           gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

}  // extern "C"
