#include "glwegadget_arithmetic.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "backend_arithmetic.h"
#include "bivariate_polynomial.h"
#include "ggsw_params.h"
#include "glwe_arithmetic.h"
#include "glwe_ciphertext.h"
#include "glwe_key.h"
#include "glwe_params.h"
#include "glwegadget_ciphertext.h"
#include "glwegadget_key.h"
#include "maths_structures.h"
#include "univariate_polynomial.h"
#include "utils.h"

int glwegadget_half_prod(const PvdaBackend* module, GLWECiphertext* result,
                         const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBiv* a)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result->params);

	// Temporaries live next to the operands (GPU memory on a CUDA backend)
	int64_t* a_tmp             = NULL;
	GLWECiphertextDFT glwe_dft = {.params = result->params, .vec = NULL};

	// `a` may be in host memory with the other operands on the GPU: its first nrows limbs
	// are then copied next to them
	PolyBiv a_view = *a;
	if (pvda_on_gpu(module, glwegadget_prep_ct->mat) && !pvda_on_gpu(module, a->ptr))
	{
		uint64_t rows = a->l < nrows ? a->l : nrows;
		a_tmp         = pvda_alloc_like(module, glwegadget_prep_ct->mat, (rows ? rows : 1) * nn * sizeof(int64_t));
		CHECK_ALLOC(a_tmp, "Allocation failed in half-product");
		for (uint64_t i = 0; i < rows; ++i)
			pvda_memcpy(module, a_tmp + i * nn, a->ptr + i * a->stride, nn * sizeof(int64_t));
		a_view = new_biv_view(nn, rows, (int64_t)nn, (PolyBivUnderlying*)a_tmp);
	}

	glwe_dft.vec = pvda_alloc_like(module, result->vec, glwe_coef_number_dft(result->params) * 2 * sizeof(double));
	CHECK_ALLOC(glwe_dft.vec, "Allocation failed in half-product");

	CHECK_CALL(pvda_vmp_apply_dft(module, glwe_dft.vec, ncols_out, &a_view, glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	CHECK_CALL(glwe_dft_to_coef(module, result, &glwe_dft),
	           "conversion from GLWE DFT to coefs failed in GLWEGadget half product");
	status = 0;
cleanup:
	pvda_free_like(module, glwe_dft.vec);
	pvda_free_like(module, a_tmp);

	return status;
}

int glwegadget_half_prod_dft_to_dft(const PvdaBackend* module, GLWECiphertextDFT* result_dft,
                                    const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBivDFT* a_dft)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result_dft->params);

	CHECK_CALL(pvda_vmp_apply_dft_to_dft(module, result_dft->vec, ncols_out, (double*)a_dft, nrows,
	                                     glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	status = 0;
cleanup:
	return status;
}

int glwegadget_half_prod_prepared_to_dft(const PvdaBackend* module, GLWECiphertextDFT* result_dft,
                                         const GLWEGadgetCiphertextPrep* glwegadget_prep_ct, const PolyBivPrep* a_prep)
{
	int status = -1;

	size_t nrows     = glwegadget_prep_ct->params->l_tilde;
	uint64_t nn      = glwegadget_prep_ct->params->params_glwe->nn;
	size_t ncols_in  = glwe_params_n_limbs(glwegadget_prep_ct->params->params_glwe);
	size_t ncols_out = glwe_params_n_limbs(result_dft->params);

	CHECK_CALL(pvda_vmp_apply_prepared_to_dft(module, result_dft->vec, ncols_out, (double*)a_prep, nrows,
	                                          glwegadget_prep_ct->mat, nrows, ncols_in),
	           "vmp apply falied in half product");

	status = 0;
cleanup:
	return status;
}

int compute_ksk(const PvdaBackend* module, GLWEKSK* ksk, const GLWESecretKeyPrepared* new_key,
                const GLWESecretKeyPrepared* old_key)
{
	int status = -1;

	uint64_t k  = new_key->k;
	uint64_t nn = new_key->nn;

	GLWEGadgetCiphertext* glwegad_tmp = new_glwegadget(ksk->params);

	CHECK_ALLOC(glwegad_tmp, "GLWEGadget allocation failed in automorphism KSK preparation");

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

int compute_automorphism_key(const PvdaBackend* module, GLWEAutomorphismKey* automorphism_key,
                             const GLWESecretKeyPrepared* glwe_key, int automorphism_p)
{
	int status = -1;

	if (!(automorphism_p & 1))
	{  //If autmorphism_p is even
		RAISE_ERROR("Cannot prepare autmorphism key for even p, operation is not well-defined");
	}

	uint64_t k  = glwe_key->k;
	uint64_t nn = glwe_key->nn;

	GLWEGadgetCiphertext* glwegad_tmp = new_glwegadget(automorphism_key->glwe_ksk->params);
	PolyUniv* auto_sk_tmp             = new_univ(automorphism_key->glwe_ksk->params->params_glwe);

	CHECK_ALLOC(glwegad_tmp, "GLWEGadget allocation failed in automorphism key computation");
	CHECK_ALLOC(auto_sk_tmp, "Allocation failed in automorphism key computation");

	automorphism_key->automorphism_p = automorphism_p;

	for (int i = 0; i < k; ++i)
	{
		GLWEGadgetCiphertextPrep* gadget_ciph = automorphism_key->glwe_ksk->enc_s[i];

		//sigma_p (sk_i)
		pvda_znx_automorphism(module, automorphism_p, auto_sk_tmp, glwe_prepared_sk_extract_poly_coefs(glwe_key, i));

		for (int i = 0; i < nn; ++i)
		{
			auto_sk_tmp[i] = -auto_sk_tmp[i];
		}

		//GLWEGadget(sigma_p(sk_i))
		CHECK_CALL(glwegadget_secret_encrypt(module, glwegad_tmp, glwe_key, auto_sk_tmp),
		           "GLWEGadget encryption failed in autmorphism key computation");

		// Prepared on the side of gadget_ciph (the GPU for a key from new_automorphism_key_on on
		// a CUDA backend), the host-encrypted gadget being uploaded if needed
		CHECK_CALL(glwegadget_prepare(module, gadget_ciph, glwegad_tmp),
		           "GLWEGadget preparation failed in automorphism key computation");
	}

	status = 0;
cleanup:
	delete_glwegadget(glwegad_tmp);
	delete_univ(auto_sk_tmp);
	return status;
}

int glwegadget_automorphism(const PvdaBackend* module, GLWECiphertext* result, const GLWEAutomorphismKey* auto_key,
                            const GLWECiphertext* glwe)
{
	int status = -1;

	uint64_t nn            = result->params->nn;
	uint64_t k             = result->params->k;
	size_t nrows           = auto_key->glwe_ksk->params->l_tilde;
	uint64_t l_b_result    = glwe_params_l_b(result->params);
	int64_t automorphism_p = auto_key->automorphism_p;

	// This is the maximum internal precision of the result.
	// It is the maximum of the input b precision and the number of columns (GLWEGaget l_tilde precision) in the
	// key-switching key
	uint64_t biv_l = l_b_result > nrows ? l_b_result : nrows;

	// Temporaries live next to the operands (GPU memory on a CUDA backend)
	PolyBiv auto_tmp_s =
	    new_biv_view(nn, biv_l, (int64_t)nn, pvda_alloc_like(module, result->vec, biv_l * nn * sizeof(int64_t)));
	PolyBiv* auto_tmp        = &auto_tmp_s;
	GLWECiphertext* glwe_tmp = NULL;
	CHECK_ALLOC(auto_tmp->ptr, "Allocation failed in automorphism");

	if (k == 1)
	{
		// auto_tmp = auto_p(a)
		PolyBiv a = glwe_extract_poly_view(glwe, 0);
		CHECK_CALL(pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a), "automorphism failed");

		// result = halfProd(C_auto(-s), auto(a)) = -halfProd(C_auto(s), a)
		CHECK_CALL(glwegadget_half_prod(module, result, auto_key->glwe_ksk->enc_s[0], auto_tmp),
		           "half product in automorphism failed");
	}
	else
	{
		glwe_tmp = new_glwe_like(module, result->params, result);
		CHECK_ALLOC(glwe_tmp, "GLWE allocation failed in automorphism");

		// auto_tmp = auto_p(a_0)
		PolyBiv a_0 = glwe_extract_poly_view(glwe, 0);
		CHECK_CALL(pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a_0), "automorphism failed");

		// result = halfProd(C_auto(-s_0), auto(a_0))
		CHECK_CALL(glwegadget_half_prod(module, result, auto_key->glwe_ksk->enc_s[0], auto_tmp),
		           "half product in automorphism failed");

		for (int i = 1; i < k; ++i)
		{
			// auto_tmp = auto_p(a_i)
			PolyBiv a_i = glwe_extract_poly_view(glwe, i);
			CHECK_CALL(pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &a_i), "automorphism failed");

			// result = halfProd(C_auto(-s_i), auto(a_i)) = -halfProd(C_auto(s_i), a_i)
			CHECK_CALL(glwegadget_half_prod(module, glwe_tmp, auto_key->glwe_ksk->enc_s[i], auto_tmp),
			           "half product in automorphism failed");

			add_glwe(module, result, result, glwe_tmp);
		}
	}

	// auto_tmp = auto_p(b)
	PolyBiv b = glwe_extract_poly_view(glwe, k);
	CHECK_CALL(pvda_vec_znx_automorphism(module, automorphism_p, auto_tmp, &b), "automorphism failed");

	// result += auto_tmp ==> result = -sum_i(halfProd(c_auto(s), auto(a_i))) + (0, auto(b))
	PolyBiv result_b = glwe_extract_poly_view(result, k);
	CHECK_CALL(pvda_vec_znx_add(module, &result_b, &result_b, auto_tmp), "addition failed in automorphism");

	status = 0;
cleanup:
	pvda_free_like(module, auto_tmp_s.ptr);
	delete_glwe(glwe_tmp);
	return status;
}

int glwe_to_glwe_keyswitch(const PvdaBackend* module, GLWECiphertext* result, const GLWEKSK* glwe_ksk,
                           const GLWECiphertext* glwe_ct)
{
	int status = -1;

	uint64_t k = result->params->k;

	if (k == 1)
	{
		PolyBiv a = glwe_extract_poly_view(glwe_ct, 0);
		//result = GLWE_k(k_new) \hp a
		//result = C_k(k_new) \hp a
		CHECK_CALL(glwegadget_half_prod(module, result, glwe_ksk->enc_s[0], &a), "half product in automorphism failed");

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
		CHECK_CALL_LABEL(glwegadget_half_prod(module, result, glwe_ksk->enc_s[0], &a_0),
		                 "half product in automorphism failed", cleanup2);

		for (int i = 1; i < k; ++i)
		{
			PolyBiv a_i = glwe_extract_poly_view(glwe_ct, i);

			//result += C_k(k_new[i]) \hp a_i
			//result = sum_{j=0}^{j=i} (C_k(k_new[j]) \hp a_j) (LOOP INVARIANT)
			CHECK_CALL_LABEL(glwegadget_half_prod(module, glwe_tmp, glwe_ksk->enc_s[i], &a_i),
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

int glwe_trace_expand(const PvdaBackend* module, GLWECiphertext** results, int res_size, const GLWECiphertext* glwe_ct,
                      const GLWEAutomorphismKeyCollection* auto_key_collection)
{
	int status = -1;

	uint64_t nn = glwe_ct->params->nn;

	// The results and the automorphism keys must live on the same side as glwe_ct
	// (checked on the first key: a collection is prepared for a single backend)
	int on_gpu                      = pvda_on_gpu(module, glwe_ct->vec);
	const GLWEAutomorphismKey* key0 = glwegadget_key_collection_get_key(auto_key_collection, nn + 1);
	int mixed                       = key0 && pvda_on_gpu(module, key0->glwe_ksk->enc_s[0]->mat) != on_gpu;
	for (int i = 0; i < res_size && !mixed; ++i) mixed = pvda_on_gpu(module, results[i]->vec) != on_gpu;
	if (mixed)
	{
		log_message(LOG_ERROR, "glwe_trace_expand: operands mix host and GPU memory");
		return -1;
	}

	glwe_copy(results[0], glwe_ct);

	// Temporaries live next to the operands (GPU memory on a CUDA backend)
	GLWECiphertext* tmp_glwe  = new_glwe_like(module, glwe_ct->params, glwe_ct);
	GLWECiphertext* tmp_glwe2 = new_glwe_like(module, glwe_ct->params, glwe_ct);
	CHECK_ALLOC(tmp_glwe, "Temp memory alloc in trace expansion failed");
	CHECK_ALLOC(tmp_glwe2, "Temp memory alloc in trace expansion failed");

	// Step 0:
	glwegadget_automorphism(module, tmp_glwe, glwegadget_key_collection_get_key(auto_key_collection, nn + 1),
	                        results[0]);

	add_glwe(module, tmp_glwe2, results[0], tmp_glwe);

	sub_glwe(module, tmp_glwe, results[0], tmp_glwe);
	PolyBiv tmp_flattened = glwe_flattened_biv(tmp_glwe);
	pvda_vec_znx_rotate(module, -1, &tmp_flattened, &tmp_flattened);
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
			GLWEAutomorphismKey* auto_key = glwegadget_key_collection_get_key(auto_key_collection, auto_p);
			CHECK_ALLOC(auto_key, "KSK retrieval failed in trace expand");
			glwegadget_automorphism(module, tmp_glwe, auto_key, results[b]);

			add_glwe(module, tmp_glwe2, results[b], tmp_glwe);

			sub_glwe(module, tmp_glwe, results[b], tmp_glwe);
			pvda_vec_znx_rotate(module, -p, &tmp_flattened, &tmp_flattened);
			normalize_glwe(module, results[b + dist], tmp_glwe);
			normalize_glwe(module, results[b], tmp_glwe2);
		}
		for (; b < p; ++b)
		{
			assert(b < res_size);
			GLWEAutomorphismKey* auto_key = glwegadget_key_collection_get_key(auto_key_collection, auto_p);
			CHECK_ALLOC(auto_key, "KSK retrieval failed in trace expand");
			glwegadget_automorphism(module, tmp_glwe, auto_key, results[b]);

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

int glwe_hom_trace(const PvdaBackend* module, GLWECiphertext* result, int enc_size, uint64_t pos,
                   const GLWECiphertext* glwe_ct, const GLWEAutomorphismKeyCollection* auto_key_collection)

{
	int status = -1;

	uint64_t nn = glwe_ct->params->nn;

	glwe_copy(result, glwe_ct);

	GLWECiphertext* tmp_glwe = new_glwe(glwe_ct->params);
	CHECK_ALLOC(tmp_glwe, "Temp memory alloc in trace expansion failed");

	PolyBiv tmp_flattened = glwe_flattened_biv(tmp_glwe);

	// Step 0:
	glwegadget_automorphism(module, tmp_glwe, glwegadget_key_collection_get_key(auto_key_collection, nn + 1), result);

	if (pos & 1)
	{
		sub_glwe(module, tmp_glwe, result, tmp_glwe);
		pvda_vec_znx_rotate(module, -1, &tmp_flattened, &tmp_flattened);
	}
	else
		add_glwe(module, tmp_glwe, result, tmp_glwe);
	pos >>= 1;

	normalize_glwe(module, result, tmp_glwe);

	// Rest of the steps
	for (uint64_t p = 2; p < enc_size; p *= 2)
	{
		int64_t auto_p                = (int64_t)nn / p + 1;
		GLWEAutomorphismKey* auto_key = glwegadget_key_collection_get_key(auto_key_collection, auto_p);
		CHECK_ALLOC(auto_key, "KSK retrieval failed in trace expand");
		glwegadget_automorphism(module, tmp_glwe, auto_key, result);

		if (pos & 1)
		{
			sub_glwe(module, tmp_glwe, result, tmp_glwe);
			pvda_vec_znx_rotate(module, -p, &tmp_flattened, &tmp_flattened);
		}
		else
			add_glwe(module, tmp_glwe, result, tmp_glwe);
		pos >>= 1;

		normalize_glwe(module, result, tmp_glwe);
	}

	status = 0;
cleanup:
	delete_glwe(tmp_glwe);
	return status;
}

int packed_glwegadget_trace_expand(const PvdaBackend* module, GLWEGadgetCiphertext** results, int res_size, int l_tilde,
                                   const GLWECiphertext* packed_glwegadget,
                                   const GLWEAutomorphismKeyCollection* auto_key_collection)

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

	CHECK_CALL(glwe_trace_expand(module, results_glwe, res_size * l_tilde, packed_glwegadget, auto_key_collection),
	           "glwegadget_trace_expand failed in a GGSW trace expansion");

	status = 0;
cleanup:
	return status;
}

int packed_glwegadget_trace_expand_prepared_single(const PvdaBackend* module, GLWEGadgetCiphertextPrep* results,
                                                   const GLWEGadgetParams* params_glwegad, int res_size, int l_tilde,
                                                   const GLWECiphertext* packed_glwegadget,
                                                   const GLWEAutomorphismKeyCollection* auto_key_collection)
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

	CHECK_CALL(packed_glwegadget_trace_expand(module, gptrs, res_size, l_tilde, packed_glwegadget, auto_key_collection),
	           "GLWEGadget trace expansion failed");

	CHECK_CALL(glwegadget_prepare(module, results, results_unprep),
	           "GLWEGadget prepareation failed in trace expansion");

	status = 0;
cleanup:
	delete_glwegadget(results_unprep);
	return status;
}

int packed_glwegadget_trace_expand_prepared(const PvdaBackend* module, GLWEGadgetCiphertextPrep** results, int res_size,
                                            int l_tilde, const GLWECiphertext* packed_glwegadget,
                                            const GLWEAutomorphismKeyCollection* auto_key_collection)
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

	CHECK_CALL(
	    packed_glwegadget_trace_expand(module, gadgets, res_size, l_tilde, packed_glwegadget, auto_key_collection),
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
