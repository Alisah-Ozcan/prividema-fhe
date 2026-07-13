#include "glwegadget_arithmetic.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bivariate_polynomial.h"
#include "ggsw_params.h"
#include "glwe_arithmetic.h"
#include "glwe_ciphertext.h"
#include "glwe_key.h"
#include "glwe_params.h"
#include "glwegadget_ciphertext.h"
#include "glwegadget_key.h"
#include "logger.h"
#include "maths_structures.h"
#include "rng.h"
#include "univariate_polynomial.h"
#include "utils.h"

#ifdef ENABLE_CUDA
#include "gpu/common/gpu_stream.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/host/normalize_host.h"
#include "gpu/host/vec_znx_arith_host.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"
#endif

int glwegadget_half_prod(const MODULE* module, GLWECiphertext* result,
                         const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBiv* a)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result->params);

#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(glwegadget_prep_ct->mat) && pvda_is_device_pointer(result->vec))
	{
		if (pvda_is_device_pointer(a->ptr))
		{
			// `a` is already device-resident (e.g. a GPU-computed automorphism
			// scratch buffer from glwegadget_automorphism) — use it directly,
			// no host round trip.
			gpu_glwegadget_half_prod_device((const int64_t*)a->ptr, (const int64_t*)glwegadget_prep_ct->mat,
			                                (int64_t*)result->vec, nn, nrows, ncols_in, ncols_out);
		}
		else
		{
			// Upload first nrows=l_tilde coef-domain limb polynomials of `a` to device
			int64_t* d_a = pvda_gpu_upload(a->ptr, (size_t)nrows * (size_t)nn);
			gpu_glwegadget_half_prod_device(d_a, (const int64_t*)glwegadget_prep_ct->mat, (int64_t*)result->vec, nn,
			                                nrows, ncols_in, ncols_out);
			pvda_gpu_free(d_a);
		}
		return 0;
	}
#endif

	GLWECiphertextDFT* glwe_dft = new_glwe_dft(result->params);
	CHECK_ALLOC(glwe_dft, "Allocation failed in half-product");

	CHECK_CALL(pvda_vmp_apply_dft(module, glwe_dft->vec, ncols_out, a, glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	CHECK_CALL(glwe_dft_to_coef(module, result, glwe_dft),
	           "conversion from GLWE DFT to coefs failed in GLWEGadget half product");
	status = 0;
cleanup:
	delete_glwe_dft(glwe_dft);

	return status;
}

int glwegadget_half_prod_dft_to_dft(const MODULE* module, GLWECiphertextDFT* result_dft,
                                    const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBivDFT* a_dft)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result_dft->params);

#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(glwegadget_prep_ct->mat) && pvda_is_device_pointer(result_dft->vec) &&
	    pvda_is_device_pointer(a_dft))
	{
		// a_dft is device pointer to nrows=l_tilde coef-domain int64 polys; NTT applied inside
		gpu_glwegadget_half_prod_ntt_device((const int64_t*)a_dft, (const int64_t*)glwegadget_prep_ct->mat,
		                                    (int64_t*)result_dft->vec, nn, nrows, ncols_in);
		return 0;
	}
#endif

	CHECK_CALL(pvda_vmp_apply_dft_to_dft(module, result_dft->vec, ncols_out, (double*)a_dft, nrows,
	                                     glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	status = 0;
cleanup:
	return status;
}

int glwegadget_half_prod_prepared_to_dft(const MODULE* module, GLWECiphertextDFT* result_dft,
                                         const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBivPrep* a_prep)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result_dft->params);

#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(glwegadget_prep_ct->mat) && pvda_is_device_pointer(result_dft->vec) &&
	    pvda_is_device_pointer(a_prep))
	{
		// a_prep is device pointer to nrows=l_tilde coef-domain int64 polys; NTT applied inside
		gpu_glwegadget_half_prod_ntt_device((const int64_t*)a_prep, (const int64_t*)glwegadget_prep_ct->mat,
		                                    (int64_t*)result_dft->vec, nn, nrows, ncols_in);
		return 0;
	}
#endif

	CHECK_CALL(pvda_vmp_apply_prepared_to_dft(module, result_dft->vec, ncols_out, (double*)a_prep, nrows,
	                                          glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	status = 0;
cleanup:
	return status;
}

int prepare_ksk(const MODULE* module, GLWEAutomorphismKSK* ksk, const GLWESecretKeyPrepared* new_key,
                const GLWESecretKeyPrepared* old_key)
{
	int status = -1;

	uint64_t k  = new_key->k;
	uint64_t nn = new_key->nn;

	GLWEGadgetCiphertext* glwegad_tmp = new_glwegadget(ksk->params);

	CHECK_ALLOC(glwegad_tmp, "GLWEGadget allocation failed in automorphism KSK preparation");

	ksk->automorphism_p = 0;

	for (int i = 0; i < k; ++i)
	{
		GLWEGadgetCiphertextPrep* gadget_ciph = ksk->enc_s[i];

		//GLWEGadget(sigma_p(sk_i))
		CHECK_CALL(
		    glwegadget_secret_encrypt(module, glwegad_tmp, new_key, glwe_prepared_sk_extract_poly_coefs(old_key, i)),
		    "GLWEGadget encryption failed in autmorphism KSK preparation");
		CHECK_CALL(glwegadget_prepare(module, gadget_ciph, glwegad_tmp),
		           "GLWEGadget preparation failed in automorphism KSK preparation");
	}

	status = 0;
cleanup:
	delete_glwegadget(glwegad_tmp);
	return status;
}
int prepare_automorphism_key(const MODULE* module, GLWEAutomorphismKSK* automorphism_ksk,
                             const GLWESecretKeyPrepared* glwe_key, int automorphism_p)
{
	int status = -1;

	if (!(automorphism_p & 1))
	{  //If autmorphism_p is even
		RAISE_ERROR("Cannot prepare autmorphism KSK for even p, operation is not well-defined");
	}

	uint64_t k  = glwe_key->k;
	uint64_t nn = glwe_key->nn;

	GLWEGadgetCiphertext* glwegad_tmp = new_glwegadget(automorphism_ksk->params);
	PolyUniv* auto_sk_tmp             = new_univ(automorphism_ksk->params->params_glwe);

	CHECK_ALLOC(glwegad_tmp, "GLWEGadget allocation failed in automorphism KSK preparation");
	CHECK_ALLOC(auto_sk_tmp, "Allocation failed in automorphism KSK preparation");

	automorphism_ksk->automorphism_p = automorphism_p;

	for (int i = 0; i < k; ++i)
	{
		GLWEGadgetCiphertextPrep* gadget_ciph = automorphism_ksk->enc_s[i];

		//sigma_p (sk_i)
		pvda_znx_automorphism(module, automorphism_p, auto_sk_tmp, glwe_prepared_sk_extract_poly_coefs(glwe_key, i));

		for (int i = 0; i < nn; ++i)
		{
			auto_sk_tmp[i] = -auto_sk_tmp[i];
		}

		//GLWEGadget(sigma_p(sk_i))
		CHECK_CALL(glwegadget_secret_encrypt(module, glwegad_tmp, glwe_key, auto_sk_tmp),
		           "GLWEGadget encryption failed in autmorphism KSK preparation");

#ifdef ENABLE_CUDA
		// gadget_ciph->mat being device-resident (see pvda_new_glwegadget_prep_device)
		// signals "NTT-prepare this KSK entry on the GPU" — CPU glwegadget_prepare
		// produces a spqlios DFT-domain matrix, GPU glwegadget_prepare_gpu an
		// NTT-domain one; same footprint, incompatible content, so we can't just
		// byte-copy the CPU-prepared result to device afterwards. Upload the raw
		// (unprepared, coefficient-domain) gadget_tmp instead and let
		// glwegadget_prepare's own device dispatch NTT-prepare it correctly.
		if (pvda_is_device_pointer(gadget_ciph->mat))
		{
			pvda_gpu_free((int64_t*)gadget_ciph->mat);  // free the placeholder ourselves —
			                                            // glwegadget_prepare only frees a stale *host* mat
			gadget_ciph->mat                  = NULL;
			int64_t* d_raw                    = pvda_glwegadget_to_device(glwegad_tmp);
			GLWEGadgetCiphertext raw_dev_view = {.params = glwegad_tmp->params, .mat = (MatBiv*)d_raw};
			CHECK_CALL(glwegadget_prepare(module, gadget_ciph, &raw_dev_view),
			           "GLWEGadget GPU preparation failed in automorphism KSK preparation");
			pvda_gpu_free(d_raw);
		}
		else
#endif
		{
			CHECK_CALL(glwegadget_prepare(module, gadget_ciph, glwegad_tmp),
			           "GLWEGadget preparation failed in automorphism KSK preparation");
		}
	}

	status = 0;
cleanup:
	delete_glwegadget(glwegad_tmp);
	delete_univ(auto_sk_tmp);
	return status;
}

int glwegadget_automorphism(const MODULE* module, GLWECiphertext* result, const GLWEAutomorphismKSK* automorphism_ksk,
                            const GLWECiphertext* glwe)
{
	int status = -1;

	uint64_t nn            = result->params->nn;
	uint64_t k             = result->params->k;
	size_t nrows           = automorphism_ksk->params->l_tilde;
	uint64_t l_b_result    = glwe_params_l_b(result->params);
	int64_t automorphism_p = automorphism_ksk->automorphism_p;

	// This is the maximum internal precision of the result.
	// It is the maximum of the input b precision and the number of columns (GLWEGaget l_tilde precision) in the
	// key-switching key
	uint64_t biv_l = l_b_result > nrows ? l_b_result : nrows;

#ifdef ENABLE_CUDA
	// automorphism_ksk->enc_s[0]->mat must also be device-resident: auto_tmp
	// below is a device buffer, and glwegadget_half_prod only takes its own
	// GPU branch (which accepts a device `a`) when glwegadget_prep_ct->mat is
	// a device pointer too — otherwise it falls back to a CPU path that
	// would dereference our device auto_tmp as host memory.
	if (pvda_is_device_pointer(glwe->vec) && pvda_is_device_pointer(result->vec) &&
	    pvda_is_device_pointer(automorphism_ksk->enc_s[0]->mat))
	{
		int status_gpu      = -1;
		int64_t* d_auto_tmp = pvda_gpu_alloc(biv_l * nn);
		int64_t* d_glwe_tmp = NULL;
		PolyBiv auto_tmp    = new_biv_view(nn, biv_l, (int64_t)nn, (PolyBivUnderlying*)d_auto_tmp);

		if (k == 1)
		{
			// auto_tmp = auto_p(a)
			PolyBiv a = glwe_extract_poly_view(glwe, 0);
			gpu_vec_znx_automorphism_device(automorphism_p, (const int64_t*)a.ptr, d_auto_tmp, nn, biv_l, (int64_t)nn,
			                                a.l, a.stride);

			// result = halfProd(C_auto(-s), auto(a)) = -halfProd(C_auto(s), a)
			CHECK_CALL_LABEL(glwegadget_half_prod(module, result, automorphism_ksk->enc_s[0], &auto_tmp),
			                 "half product in GPU automorphism failed", cleanup_gpu);

			// auto_tmp = auto_p(b)
			PolyBiv b = glwe_extract_poly_view(glwe, k);
			gpu_vec_znx_automorphism_device(automorphism_p, (const int64_t*)b.ptr, d_auto_tmp, nn, biv_l, (int64_t)nn,
			                                b.l, b.stride);

			// result += auto_tmp ==> result = -halfProc(c_auto(s), auto(a)) + (0, auto(b))
			// (result_b and auto_tmp are never aliased here, so — unlike the CPU
			// vec_znx_add_ref this mirrors — there's no "in-place tail is a no-op"
			// case to worry about: just add over their common limb count.)
			PolyBiv result_b   = glwe_extract_poly_view(result, k);
			uint64_t add_limbs = result_b.l < biv_l ? result_b.l : biv_l;
			for (uint64_t i = 0; i < add_limbs; ++i)
			{
				int64_t* row = (int64_t*)(result_b.ptr + (int64_t)i * result_b.stride);
				gpu_vec_znx_add_device(row, d_auto_tmp + (int64_t)i * nn, row, nn);
			}
		}
		else
		{
			d_glwe_tmp                  = pvda_gpu_alloc(glwe_coef_number(result->params));
			GLWECiphertext glwe_tmp_dev = {.params = result->params, .vec = (VecBiv*)d_glwe_tmp};

			// auto_tmp = auto_p(a_0)
			PolyBiv a_0 = glwe_extract_poly_view(glwe, 0);
			gpu_vec_znx_automorphism_device(automorphism_p, (const int64_t*)a_0.ptr, d_auto_tmp, nn, biv_l, (int64_t)nn,
			                                a_0.l, a_0.stride);

			// result = halfProd(C_auto(-s_0), auto(a_0))
			CHECK_CALL_LABEL(glwegadget_half_prod(module, result, automorphism_ksk->enc_s[0], &auto_tmp),
			                 "half product in GPU automorphism failed", cleanup_gpu);

			for (int i = 1; i < k; ++i)
			{
				// auto_tmp = auto_p(a_i)
				PolyBiv a_i = glwe_extract_poly_view(glwe, i);
				gpu_vec_znx_automorphism_device(automorphism_p, (const int64_t*)a_i.ptr, d_auto_tmp, nn, biv_l,
				                                (int64_t)nn, a_i.l, a_i.stride);

				// result = halfProd(C_auto(-s_i), auto(a_i)) = -halfProd(C_auto(s_i), a_i)
				CHECK_CALL_LABEL(glwegadget_half_prod(module, &glwe_tmp_dev, automorphism_ksk->enc_s[i], &auto_tmp),
				                 "half product in GPU automorphism failed", cleanup_gpu);

				add_glwe(module, result, result, &glwe_tmp_dev);
			}

			// auto_tmp = auto_p(b)
			PolyBiv b = glwe_extract_poly_view(glwe, k);
			gpu_vec_znx_automorphism_device(automorphism_p, (const int64_t*)b.ptr, d_auto_tmp, nn, biv_l, (int64_t)nn,
			                                b.l, b.stride);

			// result += auto_tmp ==> result = -sum_i(halfProd(c_auto(s), auto(a_i))) + (0, auto(b))
			PolyBiv result_b   = glwe_extract_poly_view(result, k);
			uint64_t add_limbs = result_b.l < biv_l ? result_b.l : biv_l;
			for (uint64_t i = 0; i < add_limbs; ++i)
			{
				int64_t* row = (int64_t*)(result_b.ptr + (int64_t)i * result_b.stride);
				gpu_vec_znx_add_device(row, d_auto_tmp + (int64_t)i * nn, row, nn);
			}
		}

		// gpu_vec_znx_automorphism_device already synchronizes internally
		// (its own scratch buffer needs it), but gpu_vec_znx_add_device in
		// the tail-add loop above does not (see its doc comment) — sync once
		// here, right before the internal scratch is freed and before
		// returning, to preserve glwegadget_automorphism's existing
		// synchronous contract for its many callers.
		pvda_gpu_stream_synchronize(pvda_gpu_stream_get_active());

		status_gpu = 0;
	cleanup_gpu:
		pvda_gpu_free(d_auto_tmp);
		if (d_glwe_tmp) pvda_gpu_free(d_glwe_tmp);
		return status_gpu;
	}
#endif

	if (k == 1)
	{
		PolyBiv* auto_tmp = new_biv_custom_params(result->params->nn, biv_l);
		CHECK_ALLOC(auto_tmp, "Allocation failed in automorphism");

		// auto_tmp = auto_p(a)
		PolyBiv a = glwe_extract_poly_view(glwe, 0);
		pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a);

		// result = halfProd(C_auto(-s), auto(a)) = -halfProd(C_auto(s), a)
		CHECK_CALL(glwegadget_half_prod(module, result, automorphism_ksk->enc_s[0], auto_tmp),
		           "half product in automorphism failed");

		// auto_tmp = auto_p(b)
		PolyBiv b = glwe_extract_poly_view(glwe, k);
		pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &b);

		// result += auto_tmp ==> result = -halfProc(c_auto(s), auto(a)) + (0, auto(b))
		PolyBiv result_b = glwe_extract_poly_view(result, k);
		pvda_vec_znx_add(module, &result_b, &result_b, auto_tmp);

		status = 0;
	cleanup:
		delete_biv(auto_tmp);
	}
	else
	{
		PolyBiv* auto_tmp        = new_biv_custom_params(result->params->nn, biv_l);
		GLWECiphertext* glwe_tmp = new_glwe(result->params);

		CHECK_ALLOC_LABEL(auto_tmp, "Allocation failed in automorphism", cleanup2);
		CHECK_ALLOC_LABEL(glwe_tmp, "GLWE allocation failed in automorphism", cleanup2);

		// auto_tmp = auto_p(a_0)
		PolyBiv a_0 = glwe_extract_poly_view(glwe, 0);
		pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a_0);

		// result = halfProd(C_auto(-s_0), auto(a_0))
		CHECK_CALL_LABEL(glwegadget_half_prod(module, result, automorphism_ksk->enc_s[0], auto_tmp),
		                 "half product in automorphism failed", cleanup2);

		for (int i = 1; i < k; ++i)
		{
			// auto_tmp = auto_p(a_i)
			PolyBiv a_i = glwe_extract_poly_view(glwe, i);
			pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a_i);

			// result = halfProd(C_auto(-s_i), auto(a_i)) = -halfProd(C_auto(s_i), a_i)
			CHECK_CALL_LABEL(glwegadget_half_prod(module, glwe_tmp, automorphism_ksk->enc_s[i], auto_tmp),
			                 "half product in automorphism failed", cleanup2);

			add_glwe(module, result, result, glwe_tmp);
		}

		// auto_tmp = auto_p(b)
		PolyBiv b = glwe_extract_poly_view(glwe, k);
		pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &b);

		// result += auto_tmp ==> result = -sum_i(halfProd(c_auto(s), auto(a_i))) + (0, auto(b))
		PolyBiv result_b = glwe_extract_poly_view(result, k);
		pvda_vec_znx_add(module, &result_b, &result_b, auto_tmp);
		status = 0;
	cleanup2:
		delete_biv(auto_tmp);
		delete_glwe(glwe_tmp);
	}

	return status;
}

int glwe_to_glwe_keyswitch(const MODULE* module, GLWECiphertext* result, const GLWEAutomorphismKSK* ksk,
                           const GLWECiphertext* glwe_ct)
{
	int status = -1;

	uint64_t k = result->params->k;

	if (k == 1)
	{
		PolyBiv a = glwe_extract_poly_view(glwe_ct, 0);
		//result = GLWE_k(k_new) \hp a
		//result = C_k(k_new) \hp a
		CHECK_CALL(glwegadget_half_prod(module, result, ksk->enc_s[0], &a), "half product in automorphism failed");

		// result = - (C_k(k_new) \hp a)
		negate_glwe(module, result, result);

		PolyBiv b        = glwe_extract_poly_view(glwe_ct, k);
		PolyBiv result_b = glwe_extract_poly_view(result, k);
		//result = (0, b) - result = (0, b) - (C_k(k_new) \hp a)
		pvda_vec_znx_add(module, &result_b, &result_b, &b);

		status = 0;
	cleanup:;
	}
	else
	{
		GLWECiphertext* glwe_tmp = new_glwe(result->params);

		CHECK_ALLOC_LABEL(glwe_tmp, "GLWE allocation failed in automorphism", cleanup2);

		PolyBiv a_0 = glwe_extract_poly_view(glwe_ct, 0);

		//result = GLWE_k(k_new[0]) \hp a_0
		//result = C_k(k_new[0]) \hp a_0
		CHECK_CALL_LABEL(glwegadget_half_prod(module, result, ksk->enc_s[0], &a_0),
		                 "half product in automorphism failed", cleanup2);

		for (int i = 1; i < k; ++i)
		{
			PolyBiv a_i = glwe_extract_poly_view(glwe_ct, i);

			//result += C_k(k_new[i]) \hp a_i
			//result = sum_{j=0}^{j=i} (C_k(k_new[j]) \hp a_j) (LOOP INVARIANT)
			CHECK_CALL_LABEL(glwegadget_half_prod(module, glwe_tmp, ksk->enc_s[i], &a_i),
			                 "half product in automorphism failed", cleanup2);

			add_glwe(module, result, result, glwe_tmp);
		}

		//result = -sum_{j=0}^{j=k-1} (C_k(k_new[j]) \hp a_j)
		negate_glwe(module, result, result);

		PolyBiv b        = glwe_extract_poly_view(glwe_ct, k);
		PolyBiv result_b = glwe_extract_poly_view(result, k);
		pvda_vec_znx_add(module, &result_b, &result_b, &b);
		status = 0;
	cleanup2:
		delete_glwe(glwe_tmp);
	}

	status = 0;
	return status;
}

// In-place rotate of a flattened GLWE view, dispatching to the GPU when its
// backing buffer is device-resident (see gpu_vec_znx_rotate_device — safe
// in-place since it computes into an internal scratch buffer before writing
// back). Falls back to the CPU reference otherwise.
static void glwe_rotate_flattened_inplace(const MODULE* module, int64_t p, PolyBiv* tmp_flattened)
{
#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(tmp_flattened->ptr))
	{
		gpu_vec_znx_rotate_device(p, (const int64_t*)tmp_flattened->ptr, (int64_t*)tmp_flattened->ptr,
		                          tmp_flattened->nn, tmp_flattened->l, tmp_flattened->stride, tmp_flattened->l,
		                          tmp_flattened->stride);
		return;
	}
#endif
	pvda_vec_znx_rotate(module, p, tmp_flattened, tmp_flattened);
}

#ifdef ENABLE_CUDA
// Half-product step shared by glwegadget_automorphism_gpu_batched below —
// same "a already device-resident" fast path as glwegadget_half_prod's own
// ENABLE_CUDA branch, called directly since a is always our own device
// scratch buffer here. gpu_glwegadget_half_prod_device allocates its own
// internal NTT scratch (freed before it returns) and so still synchronizes
// internally — unlike the other steps below, there is nothing to batch here
// without a further NTT-scratch-reuse refactor of that function itself.
static void glwegadget_half_prod_gpu(GLWECiphertext* result, const GLWEGadgetCiphertextPrep* glwegadget_prep_ct,
                                     const PolyBiv* a)
{
	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result->params);

	gpu_glwegadget_half_prod_device((const int64_t*)a->ptr, (const int64_t*)glwegadget_prep_ct->mat,
	                                (int64_t*)result->vec, nn, nrows, ncols_in, ncols_out);
}

// Batched counterpart of glwegadget_automorphism's GPU dispatch, used by
// glwe_trace_expand_gpu_batched's hot loop: mirrors that function's logic
// exactly (see glwegadget_automorphism above), but takes pre-allocated
// scratch buffers — reused across every automorphism call in the whole
// expansion instead of a pvda_gpu_alloc/pvda_gpu_free pair per call — and
// leans on gpu_vec_znx_automorphism_device_noalias/gpu_vec_znx_add_device
// not synchronizing on their own (no internal scratch of their own — see
// their doc comments) to queue every kernel on gpu_active_stream without an
// intermediate cudaStreamSynchronize. Only glwegadget_half_prod_gpu above
// still synchronizes (its own internal NTT scratch needs it). The caller
// synchronizes once after the whole batch completes. scratch_auto_tmp must
// hold >= biv_l*nn int64s; scratch_glwe_tmp (only touched when k>1) must
// hold glwe_coef_number(result->params) int64s.
static void glwegadget_automorphism_gpu_batched(GLWECiphertext* result, const GLWEAutomorphismKSK* automorphism_ksk,
                                                const GLWECiphertext* glwe, int64_t* scratch_auto_tmp,
                                                int64_t* scratch_glwe_tmp)
{
	uint64_t nn            = result->params->nn;
	uint64_t k             = result->params->k;
	uint64_t l_b_result    = glwe_params_l_b(result->params);
	int64_t automorphism_p = automorphism_ksk->automorphism_p;
	size_t nrows           = automorphism_ksk->params->l_tilde;
	uint64_t biv_l         = l_b_result > nrows ? l_b_result : nrows;

	PolyBiv auto_tmp = new_biv_view(nn, biv_l, (int64_t)nn, (PolyBivUnderlying*)scratch_auto_tmp);

	if (k == 1)
	{
		// auto_tmp = auto_p(a)
		PolyBiv a = glwe_extract_poly_view(glwe, 0);
		gpu_vec_znx_automorphism_device_noalias(automorphism_p, (const int64_t*)a.ptr, scratch_auto_tmp, nn, biv_l,
		                                        (int64_t)nn, a.l, a.stride);

		// result = halfProd(C_auto(-s), auto(a)) = -halfProd(C_auto(s), a)
		glwegadget_half_prod_gpu(result, automorphism_ksk->enc_s[0], &auto_tmp);

		// auto_tmp = auto_p(b)
		PolyBiv b = glwe_extract_poly_view(glwe, k);
		gpu_vec_znx_automorphism_device_noalias(automorphism_p, (const int64_t*)b.ptr, scratch_auto_tmp, nn, biv_l,
		                                        (int64_t)nn, b.l, b.stride);

		// result += auto_tmp
		PolyBiv result_b   = glwe_extract_poly_view(result, k);
		uint64_t add_limbs = result_b.l < biv_l ? result_b.l : biv_l;
		for (uint64_t i = 0; i < add_limbs; ++i)
		{
			int64_t* row = (int64_t*)(result_b.ptr + (int64_t)i * result_b.stride);
			gpu_vec_znx_add_device(row, scratch_auto_tmp + (int64_t)i * nn, row, nn);
		}
	}
	else
	{
		GLWECiphertext glwe_tmp_dev = {.params = result->params, .vec = (VecBiv*)scratch_glwe_tmp};

		// auto_tmp = auto_p(a_0)
		PolyBiv a_0 = glwe_extract_poly_view(glwe, 0);
		gpu_vec_znx_automorphism_device_noalias(automorphism_p, (const int64_t*)a_0.ptr, scratch_auto_tmp, nn, biv_l,
		                                        (int64_t)nn, a_0.l, a_0.stride);

		// result = halfProd(C_auto(-s_0), auto(a_0))
		glwegadget_half_prod_gpu(result, automorphism_ksk->enc_s[0], &auto_tmp);

		for (uint64_t i = 1; i < k; ++i)
		{
			// auto_tmp = auto_p(a_i)
			PolyBiv a_i = glwe_extract_poly_view(glwe, i);
			gpu_vec_znx_automorphism_device_noalias(automorphism_p, (const int64_t*)a_i.ptr, scratch_auto_tmp, nn,
			                                        biv_l, (int64_t)nn, a_i.l, a_i.stride);

			// result = halfProd(C_auto(-s_i), auto(a_i)) = -halfProd(C_auto(s_i), a_i)
			glwegadget_half_prod_gpu(&glwe_tmp_dev, automorphism_ksk->enc_s[i], &auto_tmp);

			// result += glwe_tmp_dev
			size_t total = glwe_coef_number(result->params);
			gpu_vec_znx_add_device((const int64_t*)result->vec, (const int64_t*)glwe_tmp_dev.vec,
			                       (int64_t*)result->vec, total);
		}

		// auto_tmp = auto_p(b)
		PolyBiv b = glwe_extract_poly_view(glwe, k);
		gpu_vec_znx_automorphism_device_noalias(automorphism_p, (const int64_t*)b.ptr, scratch_auto_tmp, nn, biv_l,
		                                        (int64_t)nn, b.l, b.stride);

		// result += auto_tmp
		PolyBiv result_b   = glwe_extract_poly_view(result, k);
		uint64_t add_limbs = result_b.l < biv_l ? result_b.l : biv_l;
		for (uint64_t i = 0; i < add_limbs; ++i)
		{
			int64_t* row = (int64_t*)(result_b.ptr + (int64_t)i * result_b.stride);
			gpu_vec_znx_add_device(row, scratch_auto_tmp + (int64_t)i * nn, row, nn);
		}
	}
}

// Batched counterpart of normalize_glwe's GPU dispatch (see normalize_glwe
// in glwe_arithmetic.c) — same per-component gpu_normalize_base2k_sized_device
// loop; that primitive doesn't synchronize on its own (no internal scratch —
// see its doc comment), so this simply doesn't add one either, unlike
// normalize_glwe's own wrapper which must to preserve its contract for its
// other callers.
static void normalize_glwe_gpu_batched(GLWECiphertext* result, const GLWECiphertext* glwe)
{
	uint64_t k     = result->params->k;
	uint64_t kappa = result->params->kappa;

	for (uint64_t j = 0; j <= k; j++)
	{
		PolyBiv aj_biv  = glwe_extract_poly_view(glwe, j);
		PolyBiv res_biv = glwe_extract_poly_view(result, j);
		gpu_normalize_base2k_sized_device((const int64_t*)aj_biv.ptr, aj_biv.l, aj_biv.stride, (int64_t*)res_biv.ptr,
		                                  res_biv.l, res_biv.stride, aj_biv.nn, (uint32_t)kappa);
	}
}

// Batched GPU implementation of glwe_trace_expand's algorithm (see the
// generic loop in glwe_trace_expand below, which this mirrors step for
// step): every automorphism/add/sub/rotate/normalize call across the whole
// res_size-1 step recursion is issued asynchronously against a handful of
// scratch buffers allocated once up front, instead of each call allocating
// its own scratch and blocking on cudaStreamSynchronize individually — on a
// MATRIX_ROWS=256-sized expansion the generic path costs on the order of a
// few thousand such round trips. Only one synchronize happens here, right
// before returning, once every op has actually been queued.
static int glwe_trace_expand_gpu_batched(GLWECiphertext** results, int res_size, const GLWECiphertext* glwe_ct,
                                         const GLWEAutomorphismKSKCollection* ksks, GLWECiphertext* tmp_glwe,
                                         GLWECiphertext* tmp_glwe2)
{
	int status = -1;

	uint64_t nn       = glwe_ct->params->nn;
	uint64_t k        = glwe_ct->params->k;
	size_t flat_total = glwe_coef_number(glwe_ct->params);

	int64_t* d_auto_tmp      = NULL;
	int64_t* d_glwe_tmp      = NULL;
	int64_t* d_rotate_scratch = NULL;

	// biv_l is constant across the whole expansion: every automorphism call
	// below shares tmp_glwe->params as its result, and every KSK drawn from
	// `ksks` shares the same l_tilde (a single GLWEGadgetParams object is used
	// to build the whole collection — see onionpir_client_phase0) — so the
	// very first KSK lookup already fixes the scratch size every later call
	// needs too.
	GLWEAutomorphismKSK* first_ksk = glwegadget_ksk_collection_get_key(ksks, nn + 1);
	CHECK_ALLOC(first_ksk, "KSK retrieval failed in batched trace expand");

	uint64_t l_b_result = glwe_params_l_b(tmp_glwe->params);
	uint64_t nrows0      = first_ksk->params->l_tilde;
	uint64_t biv_l       = l_b_result > nrows0 ? l_b_result : nrows0;

	d_auto_tmp = pvda_gpu_alloc(biv_l * nn);
	CHECK_ALLOC(d_auto_tmp, "Scratch allocation failed in batched trace expand");

	if (k > 1)
	{
		d_glwe_tmp = pvda_gpu_alloc(flat_total);
		CHECK_ALLOC(d_glwe_tmp, "Scratch allocation failed in batched trace expand");
	}

	PolyBiv tmp_flattened = glwe_flattened_biv(tmp_glwe);
	d_rotate_scratch      = pvda_gpu_alloc((size_t)tmp_flattened.l * (size_t)tmp_flattened.nn);
	CHECK_ALLOC(d_rotate_scratch, "Rotate scratch allocation failed in batched trace expand");

	// Step 0:
	glwegadget_automorphism_gpu_batched(tmp_glwe, first_ksk, results[0], d_auto_tmp, d_glwe_tmp);

	gpu_vec_znx_add_device((const int64_t*)results[0]->vec, (const int64_t*)tmp_glwe->vec, (int64_t*)tmp_glwe2->vec,
	                       flat_total);
	gpu_vec_znx_sub_device((const int64_t*)results[0]->vec, (const int64_t*)tmp_glwe->vec, (int64_t*)tmp_glwe->vec,
	                       flat_total);

	gpu_vec_znx_rotate_device_scratch(-1, (const int64_t*)tmp_flattened.ptr, (int64_t*)tmp_flattened.ptr,
	                                  tmp_flattened.nn, tmp_flattened.l, tmp_flattened.stride, tmp_flattened.l,
	                                  tmp_flattened.stride, d_rotate_scratch);

	normalize_glwe_gpu_batched(results[1], tmp_glwe);
	normalize_glwe_gpu_batched(results[0], tmp_glwe2);

	// Rest of the steps
	for (uint64_t p = 2; p < (uint64_t)res_size; p *= 2)
	{
		int64_t auto_p = (int64_t)nn / p + 1;
		int64_t dist   = p;
		uint64_t b;
		for (b = 0; b < p && b + dist < (uint64_t)res_size; ++b)
		{
			assert(b < (uint64_t)res_size);
			GLWEAutomorphismKSK* ksk = glwegadget_ksk_collection_get_key(ksks, auto_p);
			CHECK_ALLOC(ksk, "KSK retrieval failed in batched trace expand");

			glwegadget_automorphism_gpu_batched(tmp_glwe, ksk, results[b], d_auto_tmp, d_glwe_tmp);

			gpu_vec_znx_add_device((const int64_t*)results[b]->vec, (const int64_t*)tmp_glwe->vec,
			                       (int64_t*)tmp_glwe2->vec, flat_total);
			gpu_vec_znx_sub_device((const int64_t*)results[b]->vec, (const int64_t*)tmp_glwe->vec,
			                       (int64_t*)tmp_glwe->vec, flat_total);

			gpu_vec_znx_rotate_device_scratch(-(int64_t)p, (const int64_t*)tmp_flattened.ptr,
			                                  (int64_t*)tmp_flattened.ptr, tmp_flattened.nn, tmp_flattened.l,
			                                  tmp_flattened.stride, tmp_flattened.l, tmp_flattened.stride,
			                                  d_rotate_scratch);

			normalize_glwe_gpu_batched(results[b + dist], tmp_glwe);
			normalize_glwe_gpu_batched(results[b], tmp_glwe2);
		}
		for (; b < p; ++b)
		{
			assert(b < (uint64_t)res_size);
			GLWEAutomorphismKSK* ksk = glwegadget_ksk_collection_get_key(ksks, auto_p);
			CHECK_ALLOC(ksk, "KSK retrieval failed in batched trace expand");

			glwegadget_automorphism_gpu_batched(tmp_glwe, ksk, results[b], d_auto_tmp, d_glwe_tmp);

			gpu_vec_znx_add_device((const int64_t*)results[b]->vec, (const int64_t*)tmp_glwe->vec,
			                       (int64_t*)tmp_glwe->vec, flat_total);

			normalize_glwe_gpu_batched(results[b], tmp_glwe);
		}
	}

	// Every op above was only ever queued on gpu_active_stream — block once
	// here until the whole batch has actually completed, instead of after
	// each individual automorphism/add/sub/rotate/normalize like the generic
	// path below does.
	pvda_gpu_stream_synchronize(pvda_gpu_stream_get_active());

	status = 0;
cleanup:
	if (d_rotate_scratch) pvda_gpu_free(d_rotate_scratch);
	if (d_auto_tmp) pvda_gpu_free(d_auto_tmp);
	if (d_glwe_tmp) pvda_gpu_free(d_glwe_tmp);
	return status;
}
#endif

int glwe_trace_expand(const MODULE* module, GLWECiphertext** results, int res_size, const GLWECiphertext* glwe_ct,
                      const GLWEAutomorphismKSKCollection* ksks)
{
	int status = -1;

	uint64_t nn = glwe_ct->params->nn;

	glwe_copy(results[0], glwe_ct);

	GLWECiphertext* tmp_glwe;
	GLWECiphertext* tmp_glwe2;
#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(glwe_ct->vec))
	{
		// Keep the scratch buffers on-device too, so every op below
		// (glwegadget_automorphism, add_glwe, sub_glwe,
		// glwe_rotate_flattened_inplace, normalize_glwe) sees an
		// all-device operand set and takes its GPU branch.
		tmp_glwe  = pvda_new_glwe_device(glwe_ct->params);
		tmp_glwe2 = pvda_new_glwe_device(glwe_ct->params);
		// pvda_new_glwe_device leaves its buffer uninitialised (unlike
		// new_glwe's calloc below) — zero it to match.
		size_t tmp_elems = glwe_coef_number(glwe_ct->params);
		pvda_gpu_zero((int64_t*)tmp_glwe->vec, tmp_elems);
		pvda_gpu_zero((int64_t*)tmp_glwe2->vec, tmp_elems);
	}
	else
#endif
	{
		tmp_glwe  = new_glwe(glwe_ct->params);
		tmp_glwe2 = new_glwe(glwe_ct->params);
	}
	CHECK_ALLOC(tmp_glwe, "Temp memory alloc in trace expansion failed");
	CHECK_ALLOC(tmp_glwe2, "Temp memory alloc in trace expansion failed");

#ifdef ENABLE_CUDA
	if (pvda_is_device_pointer(glwe_ct->vec))
	{
		status = glwe_trace_expand_gpu_batched(results, res_size, glwe_ct, ksks, tmp_glwe, tmp_glwe2);
		goto cleanup;
	}
#endif

	// Step 0:
	glwegadget_automorphism(module, tmp_glwe, glwegadget_ksk_collection_get_key(ksks, nn + 1), results[0]);

	add_glwe(module, tmp_glwe2, results[0], tmp_glwe);

	sub_glwe(module, tmp_glwe, results[0], tmp_glwe);
	PolyBiv tmp_flattened = glwe_flattened_biv(tmp_glwe);
	glwe_rotate_flattened_inplace(module, -1, &tmp_flattened);
	normalize_glwe(module, results[1], tmp_glwe);
	normalize_glwe(module, results[0], tmp_glwe2);

	// Rest of the steps
	for (uint64_t p = 2; p < res_size; p *= 2)
	{
		int64_t auto_p = (int64_t)nn / p + 1;
		int64_t dist   = p;
		uint64_t b;
		for (b = 0; b < p && b + dist < res_size; ++b)
		{
			assert(b < res_size);
			GLWEAutomorphismKSK* ksk = glwegadget_ksk_collection_get_key(ksks, auto_p);
			CHECK_ALLOC(ksk, "KSK retrieval failed in trace expand");
			glwegadget_automorphism(module, tmp_glwe, ksk, results[b]);

			add_glwe(module, tmp_glwe2, results[b], tmp_glwe);

			sub_glwe(module, tmp_glwe, results[b], tmp_glwe);
			glwe_rotate_flattened_inplace(module, -(int64_t)p, &tmp_flattened);
			normalize_glwe(module, results[b + dist], tmp_glwe);
			normalize_glwe(module, results[b], tmp_glwe2);
		}
		for (; b < p; ++b)
		{
			assert(b < res_size);
			GLWEAutomorphismKSK* ksk = glwegadget_ksk_collection_get_key(ksks, auto_p);
			CHECK_ALLOC(ksk, "KSK retrieval failed in trace expand");
			glwegadget_automorphism(module, tmp_glwe, ksk, results[b]);

			add_glwe(module, tmp_glwe, results[b], tmp_glwe);

			normalize_glwe(module, results[b], tmp_glwe);
		}
	}

	status = 0;
cleanup:
	delete_glwe(tmp_glwe);
	delete_glwe(tmp_glwe2);
	return status;
}

int packed_glwegadget_trace_expand(const MODULE* module, GLWEGadgetCiphertext** results, int res_size, int l_tilde,
                                   const GLWECiphertext* packed_glwegadget,
                                   const GLWEAutomorphismKSKCollection* auto_ksks)

{
	int status = -1;
	// TODO: add assertion/case to check if result has l_tilde less than l_tilde itself
	// results_glwe entries are trivial wrapper views (params + a pointer already
	// owned by results[]) — plain stack storage instead of a
	// malloc/free-per-entry (res_size*l_tilde of them, e.g. 256*4=1024 for a
	// MATRIX_ROWS row-query expansion) avoids that heap churn on every query.
	GLWECiphertext results_glwe_storage[res_size * l_tilde];
	GLWECiphertext* results_glwe[res_size * l_tilde];
	int64_t k = (int64_t)packed_glwegadget->params->k;

	/*
	 * Create dummy GLWECiphertext for the k'th GLWEs in each GGSW, that is, the ones that contain a
	 * m * 2^-jK.
	 * In other words, we are storing the results as GLWEGadgets inside the GGSWs by using only
	 * one every k GLWEs in a GGSW.
	 * That way, since we fill every k'th GLWE in a GGSW, to convert this "strided" GLWEGadget into a
	 * proper GGSW, it will suffice to generate the (-sk_i * m * 2^-jK) GLWEs that we are missing,
	 * which we can do by means of an external products with encryptions of -sk_i
	 */
	for (uint64_t prec_lvl = 1; prec_lvl <= l_tilde; ++prec_lvl)
	{
		for (uint64_t res_num = 0; res_num < res_size; ++res_num)
		{
			GLWECiphertext* tmp                               = &results_glwe_storage[(prec_lvl - 1) * res_size + res_num];
			tmp->params                                       = results[res_num]->params->params_glwe;
			tmp->vec                                          = glwegadget_extract_bivglwe(results[res_num], prec_lvl);
			results_glwe[(prec_lvl - 1) * res_size + res_num] = tmp;
		}
	}

	CHECK_CALL(glwe_trace_expand(module, results_glwe, res_size * l_tilde, packed_glwegadget, auto_ksks),
	           "glwegadget_trace_expand failed in a GGSW trace expansion");

	status = 0;
cleanup:
	return status;
}

int packed_glwegadget_trace_expand_prepared_single(const MODULE* module, GLWEGadgetCiphertextPrep* results,

                                                   const GLWEGadgetParams* params_glwegad, int res_size, int l_tilde,
                                                   const GLWECiphertext* packed_glwegadget,
                                                   const GLWEAutomorphismKSKCollection* auto_ksks)
{
	int status = -1;

	GLWEGadgetCiphertext gadgets[res_size];
	GLWEGadgetCiphertext* gptrs[res_size];
	GLWEGadgetCiphertext* results_unprep = new_glwegadget(results->params);
	CHECK_ALLOC(results_unprep, "unprepared gadget allocation failed in prepared glwegadget trace expansion");
	assert(results->params->l_tilde == res_size * params_glwegad->l_tilde);
	for (int r = 0; r < res_size; ++r)
	{
		gadgets[r].params = params_glwegad;
		gadgets[r].mat    = glwegadget_extract_bivglwe(results_unprep, 1 + r * params_glwegad->l_tilde);
		gptrs[r]          = &gadgets[r];
	}

	CHECK_CALL(packed_glwegadget_trace_expand(module, gptrs, res_size, l_tilde, packed_glwegadget, auto_ksks),
	           "GLWEGadget trace expansion failed");

	CHECK_CALL(glwegadget_prepare(module, results, results_unprep),
	           "GLWEGadget prepareation failed in trace expansion");

	status = 0;
cleanup:
	delete_glwegadget(results_unprep);
	return status;
}

int packed_glwegadget_trace_expand_prepared(const MODULE* module, GLWEGadgetCiphertextPrep** results, int res_size,
                                            int l_tilde, const GLWECiphertext* packed_glwegadget,
                                            const GLWEAutomorphismKSKCollection* auto_ksks)
{
	int status = -1;

	GLWEGadgetCiphertext** gadgets = (GLWEGadgetCiphertext**)calloc(res_size, sizeof(GLWEGadgetCiphertext*));
	CHECK_ALLOC(gadgets, "unprepared gadget allocation failed in prepared glwegadget trace expansion");
	const GLWEGadgetParams* params_glwegad = results[0]->params;
	for (int r = 0; r < res_size; ++r)
	{
		gadgets[r] = new_glwegadget(params_glwegad);
		CHECK_ALLOC(gadgets[r], "GLWEGadget allocation failed in trace expansion");
	}

	CHECK_CALL(packed_glwegadget_trace_expand(module, gadgets, res_size, l_tilde, packed_glwegadget, auto_ksks),
	           "GLWEGadget trace expansion failed");

	for (int r = 0; r < res_size; ++r)
	{
		if (!results[r])
		{
			results[r] = new_glwegadget_prep(params_glwegad);
			CHECK_ALLOC(results[r], "Prepared GLWEGadget allocation failed");
		}
		CHECK_CALL(glwegadget_prepare(module, results[r], gadgets[r]),
		           "GLWEGadget preparation failed in trace expansion");
		delete_glwegadget(gadgets[r]);
	}

	free((void*)gadgets);
	return 0;
cleanup:
	if (gadgets)
	{
		for (int r = 0; r < res_size; ++r)
		{
			delete_glwegadget(gadgets[r]);
		}
	}
	free((void*)gadgets);
	return status;
}
