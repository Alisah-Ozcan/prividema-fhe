#include "ggsw_ciphertext.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend_arithmetic.h"
#include "bivariate_polynomial.h"
#include "ggsw_params.h"
#include "glwe_ciphertext.h"
#include "glwe_params.h"
#include "maths_structures.h"
#include "rng.h"
#include "univariate_polynomial.h"
#include "utils.h"

#ifdef ENABLE_CUDA
#include "gpu/host/ggsw_external_product_gpu.h"
#endif

// bivGGSW Part (begin)

GGSWCiphertext* new_ggsw(const GGSWParams* params_ggsw)
{
	// The bivGGSW does not own the GGSWParams
	GGSWCiphertext* ggsw = malloc(sizeof(GGSWCiphertext));
	CHECK_ALLOC(ggsw, "malloc in new_ggsw");

	ggsw->params = params_ggsw;

	// Initialize the bivGGSW ciphertext with 0s'
	ggsw->mat = calloc(ggsw_coef_number(params_ggsw), sizeof(int64_t));
	CHECK_ALLOC(ggsw->mat, "calloc in new_ggsw");

	return ggsw;
cleanup:

	free(ggsw);
	return NULL;
}

void delete_ggsw(GGSWCiphertext* ggsw)
{
	if (!ggsw) return;
#ifdef ENABLE_CUDA
	if (ggsw->mat && pvda_is_device_pointer(ggsw->mat))
		pvda_gpu_free((int64_t*)ggsw->mat);
	else
		free(ggsw->mat);
#else
	free(ggsw->mat);
#endif
	free(ggsw);
}

GGSWCiphertext* new_ggsw_on(const PvdaBackend* backend, const GGSWParams* params_ggsw)
{
#ifdef ENABLE_CUDA
	if (pvda_backend_type(backend) == PVDA_BACKEND_CUDA) return pvda_new_ggsw_device(params_ggsw);  // zeroed
#endif
	return new_ggsw(params_ggsw);
}

GGSWCiphertext* ggsw_clone_on(const PvdaBackend* backend, const GGSWCiphertext* src)
{
	size_t n_elems = ggsw_coef_number(src->params);
#ifdef ENABLE_CUDA
	int src_on_device = pvda_is_device_pointer(src->mat);
	if (pvda_backend_type(backend) == PVDA_BACKEND_CUDA)
	{
		if (!src_on_device) return pvda_ggsw_upload(src);
		GGSWCiphertext* dst = pvda_new_ggsw_device(src->params);
		if (dst) pvda_gpu_copy((int64_t*)dst->mat, (const int64_t*)src->mat, n_elems);
		return dst;
	}
	if (src_on_device)
	{
		GGSWCiphertext* dst = new_ggsw(src->params);
		if (dst) pvda_gpu_download((int64_t*)dst->mat, (const int64_t*)src->mat, n_elems);
		return dst;
	}
#endif
	GGSWCiphertext* dst = new_ggsw(src->params);
	if (dst) memcpy(dst->mat, src->mat, n_elems * sizeof(int64_t));
	return dst;
}

VecBiv* ggsw_retrieve_bivglwe(GGSWCiphertext* ggsw_ct, int64_t sk_idx, int64_t prec_lvl)
{
	assert(prec_lvl >= 1);

	// bivGLWE parameters
	const GLWEParams* params_glwe = ggsw_ct->params->params_glwe;

	// bivGGSW parameters
	uint64_t k_tilde = ggsw_ct->params->k_tilde;

	return ggsw_ct->mat + ((prec_lvl - 1) * (k_tilde + 1) + sk_idx) * glwe_coef_number(params_glwe);
}

PolyBiv ggsw_flattened_biv(const GGSWCiphertext* ggsw_ct)
{
	PolyBiv res = new_biv_view(ggsw_ct->params->params_glwe->nn, ggsw_total_n_glwe_limbs(ggsw_ct->params),
	                           (int64_t)ggsw_ct->params->params_glwe->nn, ggsw_ct->mat);
	return res;
}
int ggsw_secret_encrypt(const PvdaBackend* module, GGSWCiphertext* result, const GLWESecretKeyPrepared* sk_prep,
                        const PolyUniv* m_univ)
{
	int status = -1;

	const GGSWParams* params_ggsw = result->params;
	const GLWEParams* params_glwe = params_ggsw->params_glwe;
	if (params_ggsw->k_tilde > params_ggsw->params_glwe->k)
		return log_perror("k_tilde should not be greater than k in ggsw_secret_encrypt");

	// bivGLWE parameters
	uint64_t nn      = params_glwe->nn;
	uint64_t k       = params_glwe->k;
	uint64_t k_tilde = params_ggsw->k_tilde;

	// Variables
	PolyUnivDFT* m_univ_dft     = new_univ_dft(module);   // DFT(msg)
	PolyUnivDFT* m_skj_univ_dft = new_univ_dft(module);   // DFT(msg * sk_j)
	PolyUniv* m_skj_univ        = new_univ(params_glwe);  // -msg * sk_j
	// Temp space for -m * sk * 2^{-kappa_tilde}
	PolyBiv* glwe_biv_msg = new_biv(params_glwe);

	CHECK_ALLOC(m_univ_dft, "malloc failed in ggsw_secret_encrypt");
	CHECK_ALLOC(m_skj_univ_dft, "malloc failed in ggsw_secret_encrypt");
	CHECK_ALLOC(m_skj_univ, "malloc failed in ggsw_secret_encrypt");
	CHECK_ALLOC(glwe_biv_msg, "malloc failed in ggsw_secret_encrypt");

	// Computes DFT(msg)
	univ_coefs_to_dft(module, m_univ_dft, m_univ);

	for (uint64_t ij = 0; ij < ggsw_num_rows(params_ggsw); ++ij)
	{
		uint64_t j = (ij % (k + 1));
		uint64_t i = ij / (k + 1) + 1;
		{
			// For j < k, m_skj_univ is -m * skj
			// For j = k  we skip this and we get it from m_univ directly
			if (j < params_glwe->k)
			{
				// Computes DFT(msg * sk_j)
				pvda_svp_apply_dft_to_dft(module, m_skj_univ_dft, 1, glwe_prepared_sk_extract_poly_dft(sk_prep, j),
				                          m_univ_dft, 1);

				// Computes -DFT(msg * sk_j)
				for (uint64_t p = 0; p < nn; p++) m_skj_univ_dft[p] = -1 * m_skj_univ_dft[p];

				// Invert the DFT to get -msg * sk_j
				CHECK_CALL(univ_dft_to_coefs(module, m_skj_univ, m_skj_univ_dft),
				           "vec_znx_idft_p failed in compute_phase_ij");
			}

			CHECK_CALL(univ_znx_to_biv(params_glwe, glwe_biv_msg, (k == j) ? m_univ : m_skj_univ,
			                           params_ggsw->kappa_tilde * i),
			           "univ_to_biv failed in compute_phase_ij");

			CHECK_CALL(add_biv_noise(module, params_glwe, glwe_biv_msg, glwe_biv_msg),
			           "Noise addition failed in GGSW encryption");
		}
		// Get the pointer for the result position
		VecBiv* glwe_vec       = ggsw_retrieve_bivglwe(result, j, i);
		GLWECiphertext glwe_ct = {params_glwe, glwe_vec};

		//Compute: bivGLWE(glwe_biv_msg) into glwe_vec
		CHECK_CALL(glwe_secret_encrypt_phase(module, &glwe_ct, sk_prep, glwe_biv_msg),
		           "glwe_secret_masking_ggsw_lib failed in ggsw_secret_encrypt");
	}

	status = 0;

cleanup:
	delete_biv(glwe_biv_msg);
	delete_univ_dft(module, m_skj_univ_dft);
	delete_univ(m_skj_univ);
	delete_univ_dft(module, m_univ_dft);

	return status;
}
// bivGGSW DFT PART (begin)

GGSWCiphertextPrep* new_ggsw_prep(const GGSWParams* params_ggsw)
{
	GGSWCiphertextPrep* ggsw_mat_dft = malloc(sizeof(GGSWCiphertextPrep));
	CHECK_ALLOC(ggsw_mat_dft, "malloc in new_ggsw_dft");

	ggsw_mat_dft->params = params_ggsw;

	// Initializes  the bivGGSW ciphertext with Os'
	ggsw_mat_dft->mat = calloc(2 * ggsw_coef_number_dft(params_ggsw), sizeof(double));
	CHECK_ALLOC(ggsw_mat_dft->mat, "calloc in new_ggsw_dft");

	return ggsw_mat_dft;
cleanup:
	delete_ggsw_prep(ggsw_mat_dft);
	return NULL;
}

GGSWCiphertextPrep* new_ggsw_prep_on(const PvdaBackend* backend, const GGSWParams* params_ggsw)
{
#ifdef ENABLE_CUDA
	if (pvda_backend_type(backend) == PVDA_BACKEND_CUDA) return pvda_new_ggsw_prep_device(params_ggsw);
#endif
	return new_ggsw_prep(params_ggsw);
}

void delete_ggsw_prep(GGSWCiphertextPrep* ggsw_dft)
{
	if (!ggsw_dft) return;
#ifdef ENABLE_CUDA
	if (ggsw_dft->mat && pvda_is_device_pointer(ggsw_dft->mat))
		pvda_gpu_free((int64_t*)ggsw_dft->mat);
	else
		free(ggsw_dft->mat);
#else
	free(ggsw_dft->mat);
#endif
	free(ggsw_dft);
}

int ggsw_prepare(const PvdaBackend* module, GGSWCiphertextPrep* ggsw_prepared, const GGSWCiphertext* ggsw_ct)
{
	int status = -1;

	size_t nrows = ggsw_prepared->params->ciphertext_nb_limbs_tilde;
	size_t ncols = glwe_params_n_limbs(ggsw_prepared->params->params_glwe);

	// The prepared ciphertext follows the raw one: (re)allocated next to it when missing or
	// on the other side (e.g. a new_ggsw_prep prepared from a GPU-resident GGSW)
	if (!ggsw_prepared->mat || pvda_on_gpu(module, ggsw_prepared->mat) != pvda_on_gpu(module, ggsw_ct->mat))
	{
		pvda_free_like(module, ggsw_prepared->mat);
		ggsw_prepared->mat =
		    pvda_alloc_like(module, ggsw_ct->mat, 2 * ggsw_coef_number_dft(ggsw_prepared->params) * sizeof(double));
		CHECK_ALLOC(ggsw_prepared->mat, "Prepared GGSW allocation failed in ggsw_prepare");
	}

	CHECK_CALL(pvda_vmp_prepare_contiguous(module, ggsw_prepared->mat, ggsw_ct->mat, nrows, ncols),
	           "VMP prepare for GLWEGadget prepare failed");

	status = 0;
cleanup:
	return status;
}
