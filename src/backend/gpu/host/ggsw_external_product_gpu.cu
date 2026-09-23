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
	return pvda_gpu_guard([&]() -> int64_t* {
		Data64s* d_ptr = (Data64s*)pvda_device_alloc_bytes(n_elements * sizeof(int64_t));
		CUDA_CHECK(
		    cudaMemcpyAsync(d_ptr, host_ptr, n_elements * sizeof(int64_t), cudaMemcpyHostToDevice, gpu_active_stream));
		return (int64_t*)d_ptr;
	});
}

void pvda_gpu_download(int64_t* host_ptr, const int64_t* device_ptr, size_t n_elements)
{
	return pvda_gpu_guard([&]() -> void {
		// Downloads cross back into host memory that callers read immediately on
		// return (no further stream sync at the call site), so this primitive
		// keeps its synchronous contract despite using the active stream.
		CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, n_elements * sizeof(int64_t), cudaMemcpyDeviceToHost,
		                           gpu_active_stream));
		CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
	});
}

int64_t* pvda_gpu_alloc(size_t n_elements) { return (int64_t*)pvda_device_alloc_bytes(n_elements * sizeof(int64_t)); }

void pvda_gpu_free(int64_t* device_ptr) { pvda_device_free_bytes(device_ptr); }

void pvda_gpu_copy(int64_t* dst, const int64_t* src, size_t n_elements)
{
	return pvda_gpu_guard([&]() -> void {
		CUDA_CHECK(
		    cudaMemcpyAsync(dst, src, n_elements * sizeof(int64_t), cudaMemcpyDeviceToDevice, gpu_active_stream));
	});
}

void pvda_gpu_zero(int64_t* dst, size_t n_elements)
{
	return pvda_gpu_guard(
	    [&]() -> void { CUDA_CHECK(cudaMemsetAsync(dst, 0, n_elements * sizeof(int64_t), gpu_active_stream)); });
}

int64_t* pvda_glwe_to_device(const GLWECiphertext* glwe)
{
	return pvda_gpu_guard([&]() -> int64_t* {
		size_t n_elems = (size_t)glwe->params->ciphertext_nb_limbs * (size_t)glwe->params->nn;
		return pvda_gpu_upload((const int64_t*)glwe->vec, n_elems);
	});
}

void pvda_glwe_from_device(GLWECiphertext* result, const int64_t* d_data)
{
	return pvda_gpu_guard([&]() -> void {
		size_t n_elems = (size_t)result->params->ciphertext_nb_limbs * (size_t)result->params->nn;
		pvda_gpu_download((int64_t*)result->vec, d_data, n_elems);
	});
}

int64_t* pvda_ggsw_to_device(const GGSWCiphertext* ggsw)
{
	return pvda_gpu_guard([&]() -> int64_t* {
		// nrows * ncols * nn  =  ciphertext_nb_limbs_tilde * params_glwe->ciphertext_nb_limbs * params_glwe->nn
		size_t n_elems = (size_t)ggsw->params->ciphertext_nb_limbs_tilde *
		                 (size_t)ggsw->params->params_glwe->ciphertext_nb_limbs * (size_t)ggsw->params->params_glwe->nn;
		return pvda_gpu_upload((const int64_t*)ggsw->mat, n_elems);
	});
}

GLWECiphertext* pvda_new_glwe_device(const GLWEParams* params)
{
	return pvda_gpu_guard([&]() -> GLWECiphertext* {
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
	});
}

GLWEGadgetCiphertext* pvda_new_glwegadget_device(const GLWEGadgetParams* params)
{
	return pvda_gpu_guard([&]() -> GLWEGadgetCiphertext* {
		GLWEGadgetCiphertext* glwegad = (GLWEGadgetCiphertext*)malloc(sizeof(GLWEGadgetCiphertext));
		if (!glwegad) return nullptr;

		// l_tilde * ciphertext_nb_limbs * nn == glwegadget_coef_number(params) — computed
		// directly rather than calling glwegadget_coef_number, same extern "C" mangling
		// reason as pvda_new_glwe_device above.
		size_t n_elems = (size_t)params->l_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs *
		                 (size_t)params->params_glwe->nn;

		glwegad->params = params;
		glwegad->mat    = (MatBiv*)pvda_gpu_alloc(n_elems);
		pvda_gpu_zero((int64_t*)glwegad->mat, n_elems);
		return glwegad;
	});
}

GGSWCiphertext* pvda_new_ggsw_device(const GGSWParams* params)
{
	return pvda_gpu_guard([&]() -> GGSWCiphertext* {
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
	});
}

GLWECiphertext* pvda_glwe_upload(const GLWECiphertext* host_glwe)
{
	return pvda_gpu_guard([&]() -> GLWECiphertext* {
		GLWECiphertext* glwe = (GLWECiphertext*)malloc(sizeof(GLWECiphertext));
		if (!glwe) return nullptr;
		glwe->params = host_glwe->params;
		glwe->vec    = (VecBiv*)pvda_glwe_to_device(host_glwe);
		return glwe;
	});
}

void pvda_glwe_download(GLWECiphertext* host_dst, const GLWECiphertext* device_glwe)
{
	return pvda_gpu_guard([&]() -> void { pvda_glwe_from_device(host_dst, (const int64_t*)device_glwe->vec); });
}

GGSWCiphertext* pvda_ggsw_upload(const GGSWCiphertext* host_ggsw)
{
	return pvda_gpu_guard([&]() -> GGSWCiphertext* {
		GGSWCiphertext* ggsw = (GGSWCiphertext*)malloc(sizeof(GGSWCiphertext));
		if (!ggsw) return nullptr;
		ggsw->params = host_ggsw->params;
		ggsw->mat    = (MatBiv*)pvda_ggsw_to_device(host_ggsw);
		return ggsw;
	});
}

GLWEGadgetCiphertext* pvda_glwegadget_upload(const GLWEGadgetCiphertext* host_glwegad)
{
	return pvda_gpu_guard([&]() -> GLWEGadgetCiphertext* {
		GLWEGadgetCiphertext* glwegad = (GLWEGadgetCiphertext*)malloc(sizeof(GLWEGadgetCiphertext));
		if (!glwegad) return nullptr;
		glwegad->params = host_glwegad->params;
		glwegad->mat    = (MatBiv*)pvda_glwegadget_to_device(host_glwegad);
		return glwegad;
	});
}

GLWECiphertextDFT* pvda_new_glwe_dft_device(const GLWEParams* params)
{
	return pvda_gpu_guard([&]() -> GLWECiphertextDFT* {
		GLWECiphertextDFT* glwe_dft = (GLWECiphertextDFT*)malloc(sizeof(GLWECiphertextDFT));
		if (!glwe_dft) return nullptr;
		// NTT domain: one int64 residue per coefficient (see vtable_ops_host.h)
		size_t n_elems   = (size_t)params->ciphertext_nb_limbs * (size_t)params->nn;
		glwe_dft->params = params;
		glwe_dft->vec    = (VecBivDFT*)pvda_gpu_alloc(n_elems);
		return glwe_dft;
	});
}

GLWEGadgetCiphertextPrep* pvda_new_glwegadget_prep_device(const GLWEGadgetParams* params)
{
	return pvda_gpu_guard([&]() -> GLWEGadgetCiphertextPrep* {
		GLWEGadgetCiphertextPrep* prep = (GLWEGadgetCiphertextPrep*)malloc(sizeof(GLWEGadgetCiphertextPrep));
		if (!prep) return nullptr;

		// NTT-domain prepared matrix, one int64 residue per coefficient: glwegadget_prepare
		// fills it on the GPU (see pvda_vmp_prepare_contiguous_placed)
		size_t n_elems = (size_t)params->l_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs *
		                 (size_t)params->params_glwe->nn;

		prep->params = params;
		prep->mat    = (MatBivDFT*)pvda_gpu_alloc(n_elems);
		return prep;
	});
}

GGSWCiphertextPrep* pvda_new_ggsw_prep_device(const GGSWParams* params)
{
	return pvda_gpu_guard([&]() -> GGSWCiphertextPrep* {
		GGSWCiphertextPrep* prep = (GGSWCiphertextPrep*)malloc(sizeof(GGSWCiphertextPrep));
		if (!prep) return nullptr;

		size_t n_elems = (size_t)params->ciphertext_nb_limbs_tilde * (size_t)params->params_glwe->ciphertext_nb_limbs *
		                 (size_t)params->params_glwe->nn;

		prep->params = params;
		prep->mat    = (MatBivDFT*)pvda_gpu_alloc(n_elems);
		return prep;
	});
}

int64_t* pvda_glwegadget_to_device(const GLWEGadgetCiphertext* glwegad)
{
	return pvda_gpu_guard([&]() -> int64_t* {
		size_t n_elems = (size_t)glwegad->params->l_tilde * (size_t)glwegad->params->params_glwe->ciphertext_nb_limbs *
		                 (size_t)glwegad->params->params_glwe->nn;
		return pvda_gpu_upload((const int64_t*)glwegad->mat, n_elems);
	});
}

}  // extern "C"
