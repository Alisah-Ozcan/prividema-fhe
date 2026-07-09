#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bivariate_polynomial.h"
#include "ggsw_arithmetic.h"
#include "ggsw_ciphertext.h"
#include "ggsw_key.h"
#include "ggsw_params.h"
#include "glwe_ciphertext.h"
#include "glwe_key.h"
#include "glwe_params.h"
#include "glwe_transform_key.h"
#include "glwegadget_arithmetic.h"
#include "glwegadget_ciphertext.h"
#include "glwegadget_key.h"
#include "maths_structures.h"
#include "schemes/tfhe.h"
#include "spqlios_alias.h"
#include "univariate_polynomial.h"
#include "utils.h"

#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"

/*****************************************************************************
 * OnionPIR example using Half-products — GPU server.
 *
 * Same protocol as exemples_PIR.c, but the server-side work is moved to the
 * GPU. It reuses the exact same building blocks (packed_glwegadget_trace_expand,
 * glwegadget_half_prod_prepared_to_dft, glwe_dft_to_coef, tfhe_cmux_tree) — all
 * of those already dispatch to their GPU kernels whenever they're handed CUDA
 * device pointers (see pvda_is_device_pointer checks throughout the core lib).
 *
 * Query expansion (unpacking the client's packed row/column selectors) still
 * runs on the CPU: it relies on automorphisms/rotations
 * (pvda_vec_znx_automorphism / pvda_vec_znx_rotate), which have no GPU kernel
 * in this codebase. Its output (a small raw GLWEGadget / a handful of raw
 * GGSWs) is then uploaded once per query and NTT-prepared on the GPU through
 * the existing glwegadget_prepare / ggsw_prepare dispatch, exactly like
 * ggsw_prepare_gpu is used elsewhere.
 *
 * The actual bottleneck — the 256 column half-products and the 256-leaf CMux
 * tree — runs entirely on the GPU: the database columns are precomputed once
 * (uploaded raw, un-NTT'd — the GPU half-product kernel does the NTT itself
 * on every call) and reused, read-only, across every future query.
 *
 * See:
 * OnionPIR: https://eprint.iacr.org/2021/1081
 * OnionPIRv2: https://eprint.iacr.org/2025/1142
 *
 *****************************************************************************/

// Matrix dimenstions (number of columns, rows and logarithm of the columns)
#define MATRIX_COLS 256
#define LOG2_COLS   8
#define MATRIX_ROWS 256

// How many columns to actually keep in device memory (see exemples_PIR.c —
// same in-memory-database-subset caveat, just uploaded to the GPU instead).
#define IN_MEMORY_DFT_COLS 64

// GLWE(and Gadget) parameters
#define NBASE      (1 << 12)
#define KBASE      1
#define KAPPABASE  18
#define L_TILDE_Q1 4

// See exemples_PIR.c for a full explanation of this margin.
#define SHFT_AMT 10

//Global vairable for debugging
GLWESecretKeyPrepared* dbg_key = NULL;

// Function that generates the data for a certain matrix position
// (identical to exemples_PIR.c — plaintext-like placeholder data generation,
// no GPU involvement needed here).
int onionpir_fill_bivariate_with_matrix_position(const GLWEParams* params_glwe, PolyBiv* biv, int64_t row,
                                                 int64_t column)
{
	PolyUnivTnX* test = new_univ_tnx(params_glwe);
	uint64_t rn       = (row + 1l) << SHFT_AMT;
	uint64_t cn       = (column + 1l) << SHFT_AMT;

	for (int i = 0; i < NBASE; ++i)
	{
		if (i % 2)
			test[i] = cn;
		else
			test[i] = rn;
	}
	univ_tnx_to_biv(params_glwe, biv, (uint64_t*)test, 0);

	delete_univ_tnx(test);
	return 0;
}

// Creates a matrix-column bivariate polynomial by generating each of its elements and concatenating them
int onionpir_fill_column_with_matrix_position(const GLWEParams* params_glwe, PolyBiv* biv, int64_t column,
                                              int64_t biv_depth, int64_t rows)
{
	assert(biv->l == biv_depth * rows);

	for (int i = 0; i < rows; ++i)
	{
		PolyBiv rowbiv = {biv->nn, biv_depth, biv->nn, biv->ptr + (biv_depth * biv->nn) * i};
		onionpir_fill_bivariate_with_matrix_position(params_glwe, &rowbiv, i, column);
	}
}

// Precomputed database columns, uploaded to the GPU once and reused (read-only)
// across every query. Unlike columns_dft[] in exemples_PIR.c, these hold RAW
// (un-NTT'd) coefficient-domain data — glwegadget_half_prod_prepared_to_dft's
// GPU path performs the NTT itself on every call (see gpu_glwegadget_half_prod_ntt_device),
// so there is no separate "prepare" step to do ahead of time on the vector side.
// columns_gpu[0] is a zeroed placeholder for columns beyond IN_MEMORY_DFT_COLS.
int64_t* columns_gpu[IN_MEMORY_DFT_COLS + 1] = {0};

int64_t* onionpir_get_prepared_column_gpu(int64_t column)
{
	if (column >= IN_MEMORY_DFT_COLS) return columns_gpu[0];
	return columns_gpu[column + 1];
}

// GPU counterpart of prepare_column in exemples_PIR.c: generates the raw column
// data on the CPU (as before — this is plaintext-like data, no benefit from the
// GPU here) and uploads it once to device memory.
int prepare_column_gpu(const MODULE* module, int64_t column, const GLWEParams* db_params,
                       const GLWEGadgetParams* query1_params)
{
	assert(query1_params->l_tilde == glwe_params_l_a(db_params));
	uint64_t total_depth = query1_params->l_tilde * MATRIX_ROWS;

	if (column >= IN_MEMORY_DFT_COLS)
	{
		if (columns_gpu[0] == NULL)
		{
			columns_gpu[0] = pvda_gpu_alloc(total_depth * db_params->nn);
			pvda_gpu_zero(columns_gpu[0], total_depth * db_params->nn);
		}
		return 0;
	}

	PolyBiv* pos_biv = new_biv_custom_params(db_params->nn, total_depth);
	onionpir_fill_column_with_matrix_position(db_params, pos_biv, column, query1_params->l_tilde, MATRIX_ROWS);

	columns_gpu[column + 1] = pvda_gpu_upload(pos_biv->ptr, total_depth * db_params->nn);

	delete_biv(pos_biv);
	return 0;
}

// GPU counterpart of onionpir_server in exemples_PIR.c.
int onionpir_server_gpu(const MODULE* module, const GGSWParams* ggsw_ksk_params, const GLWEGadgetParams* query1_params,
                        const GLWEParams* db_params, const GLWEParams* aggregation_params,
                        const GLWEAutomorphismKSKCollection* ksks, const GGSWCiphertextPrep** ggsw_ksks,
                        GLWECiphertext* res, const GLWECiphertext* row_query, const GLWECiphertext* col_query)
{
	// CMux-tree leaves — device-resident, one per column.
	GLWECiphertext* glwe_tree_first_level[MATRIX_COLS] = {0};
	for (int c = 0; c < MATRIX_COLS; ++c) glwe_tree_first_level[c] = pvda_new_glwe_device(aggregation_params);

	GLWEGadgetParams* mega_params = new_glwegadget_params(query1_params->params_glwe, query1_params->kappa_tilde,
	                                                      query1_params->l_tilde * MATRIX_ROWS);

	// --- Row query expansion: CPU-only (automorphisms/rotations have no GPU
	// kernel here), then upload the raw result once and NTT-prepare it on the
	// GPU through the existing glwegadget_prepare dispatch. ---
	GLWEGadgetCiphertext gadgets[MATRIX_ROWS];
	GLWEGadgetCiphertext* gptrs[MATRIX_ROWS];
	GLWEGadgetCiphertext* row_trace_unprep = new_glwegadget(mega_params);
	for (int r = 0; r < MATRIX_ROWS; ++r)
	{
		gadgets[r].params = query1_params;
		gadgets[r].mat    = glwegadget_extract_bivglwe(row_trace_unprep, 1 + r * query1_params->l_tilde);
		gptrs[r]          = &gadgets[r];
	}
	packed_glwegadget_trace_expand(module, gptrs, MATRIX_ROWS, L_TILDE_Q1, row_query, ksks);

	int64_t* d_row_trace_raw               = pvda_glwegadget_to_device(row_trace_unprep);
	GLWEGadgetCiphertext gpu_row_trace_raw = {.params = mega_params, .mat = (MatBiv*)d_row_trace_raw};
	GLWEGadgetCiphertextPrep* glwegad_trace = new_glwegadget_prep(mega_params);
	glwegadget_prepare(module, glwegad_trace, &gpu_row_trace_raw);

	pvda_gpu_free(d_row_trace_raw);
	delete_glwegadget(row_trace_unprep);

	// --- Half products (the main performance bottleneck) — fully on the GPU. ---
	GLWECiphertextDFT tmp_glwe_dft_gpu = {
	    .params = aggregation_params,
	    .vec = (VecBivDFT*)pvda_gpu_alloc((size_t)glwe_params_n_limbs(aggregation_params) * (size_t)aggregation_params->nn)};

	struct timespec server_start;
	clock_gettime(CLOCK_REALTIME, &server_start);

	for (int64_t c = 0; c < MATRIX_COLS; ++c)
	{
		glwegadget_half_prod_prepared_to_dft(module, &tmp_glwe_dft_gpu, glwegad_trace,
		                                     (const PolyBivPrep*)onionpir_get_prepared_column_gpu(c));
		glwe_dft_to_coef(module, glwe_tree_first_level[c], &tmp_glwe_dft_gpu);
	}

	struct timespec server_end;
	clock_gettime(CLOCK_REALTIME, &server_end);
	double ms_elapsed =
	    (server_end.tv_sec - server_start.tv_sec) * 1000 + (server_end.tv_nsec - server_start.tv_nsec) / 1000000;
	printf("HP (GPU) elapsed time: %.2f ms\n", ms_elapsed);

	pvda_gpu_free((int64_t*)tmp_glwe_dft_gpu.vec);
	delete_glwegadget_prep(glwegad_trace);

	// --- Column-selection query expansion: also CPU-only internally
	// (automorphisms), then upload + NTT-prepare each resulting GGSW on the GPU
	// through the existing ggsw_prepare dispatch. ---
	GGSWCiphertext* ggsw_trace_raw[LOG2_COLS] = {0};
	for (int r = 0; r < LOG2_COLS; ++r) ggsw_trace_raw[r] = new_ggsw(ggsw_ksk_params);

	packed_glwegadget_trace_expand_ggsw(module, ggsw_trace_raw, LOG2_COLS, ggsw_params_l_tilde_a(ggsw_ksk_params),
	                                    col_query, ksks, ggsw_ksks);

	GGSWCiphertextPrep* ggsw_trace[LOG2_COLS] = {0};
	for (int r = 0; r < LOG2_COLS; ++r)
	{
		int64_t* d_ggsw_raw         = pvda_ggsw_to_device(ggsw_trace_raw[r]);
		GGSWCiphertext gpu_ggsw_raw = {.params = ggsw_ksk_params, .mat = (VecBiv*)d_ggsw_raw};
		ggsw_trace[r]               = new_ggsw_prep(ggsw_ksk_params);
		ggsw_prepare(module, ggsw_trace[r], &gpu_ggsw_raw);
		pvda_gpu_free(d_ggsw_raw);
		delete_ggsw(ggsw_trace_raw[r]);
	}

	// --- CMux selection tree — entirely on the GPU: leaves, selectors and the
	// result are all device-resident, and tfhe_cmux_tree allocates its own
	// intermediate nodes on the device too (see tfhe_cmux_tree_new_node). ---
	GLWECiphertext* res_gpu = pvda_new_glwe_device(res->params);
	tfhe_cmux_tree(module, res_gpu, (const GLWECiphertext**)glwe_tree_first_level, MATRIX_COLS,
	              (const GGSWCiphertextPrep**)ggsw_trace, LOG2_COLS, 1);

	pvda_glwe_from_device(res, res_gpu->vec);
	delete_glwe(res_gpu);

	for (int i = 0; i < LOG2_COLS; ++i) delete_ggsw_prep(ggsw_trace[i]);
	delete_glwegadget_params(mega_params);
	return 0;
}

// Setup phase for the client in the protocol: secret and evaluation key generation
// (identical to exemples_PIR.c — client-side, stays on the CPU).
int onionpir_client_phase0(MODULE* module, GLWESecretKeyPrepared** sk_prep_out, int sk_bits, GLWEParams* sk_params,
                           GLWEAutomorphismKSKCollection** ksks_out, const GLWEGadgetParams* auto_ksk_params,
                           GGSWCiphertextPrep*** ggsw_ksks_out, const GGSWParams* auto_ggsw_params)
{
	int status                          = -1;
	GLWESecretKey* sk                   = alloc_glwe_secret_key(sk_params);
	GLWESecretKeyPrepared* sk_prep      = alloc_glwe_secret_key_prepared(sk_params);
	GLWEAutomorphismKSKCollection* ksks = new_automorphism_ksk_collection(2ul * NBASE);
	GGSWCiphertextPrep** ggsw_ksks      = (GGSWCiphertextPrep**)calloc(KBASE, sizeof(GGSWCiphertextPrep*));
	CHECK_ALLOC(sk, "Secret key allocation failed in phase0 of OnionPIR");
	CHECK_ALLOC(sk_prep, "Prepared secret key allocation failed in phase0 of OnionPIR");
	CHECK_ALLOC(ksks, "Allocation failed in phase0 of onionPIR");
	CHECK_ALLOC(ggsw_ksks, "Allocation failed in phase 0 of onionPIR");

	*sk_prep_out = sk_prep;
	dbg_key      = sk_prep;
	uniform_glwe_secret_key(module, sk, sk_bits);
	glwe_sk_prepare(module, sk_prep, sk);

	*ksks_out = ksks;
	for (uint64_t i = 1; (1ULL << i) <= NBASE; ++i)
	{
		int64_t p                = (int64_t)NBASE / (1LL << (i - 1)) + 1;
		GLWEAutomorphismKSK* ksk = new_automorphism_ksk(auto_ksk_params);
		prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	*ggsw_ksks_out = ggsw_ksks;
	generate_glwegad_to_ggsw_ksk(module, ggsw_ksks, auto_ggsw_params, sk_prep);

	status = 0;
cleanup:
	delete_glwe_secret_key(sk);
	return status;
}

// Initial client phase: generate the row and column packed GLWEGadgets according to the
// desired row and column to select (identical to exemples_PIR.c).
int onionpir_client_phase1(const MODULE* module, GLWECiphertext** row_query, GLWECiphertext** col_query,
                           const GLWESecretKeyPrepared* sk_prep, int row, int column,
                           const GLWEParams* params_row_query, const GLWEParams* params_col_query,
                           const GLWEGadgetParams* row_query_gad_params, const GLWEGadgetParams* col_query_gad_params)
{
	int status        = -1;
	*row_query        = new_glwe(params_row_query);
	*col_query        = new_glwe(params_col_query);
	PolyUniv* sel_row = new_univ(params_row_query);
	PolyUniv* sel_col = new_univ(params_col_query);

	memset(sel_row, 0, poly_univ_bytes(params_row_query));
	memset(sel_col, 0, poly_univ_bytes(params_col_query));

	uint64_t col_num = column;
	uint64_t row_num = row;

	for (int i = 0; i < LOG2_COLS; ++i)
	{
		sel_col[i] = (int64_t)col_num % 2;
		col_num >>= 1;
	}

	sel_row[row_num] = 1;

	glwegadget_packed_secret_encrypt(module, *row_query, row_query_gad_params, sk_prep, sel_row, MATRIX_ROWS);
	glwegadget_packed_secret_encrypt(module, *col_query, col_query_gad_params, sk_prep, sel_col, LOG2_COLS);

	status = 0;
cleanup:
	delete_univ(sel_row);
	delete_univ(sel_col);
	return status;
}

int main(int argc, char* argv[])
{
	if (argc != 1 && argc != 3)
	{
		printf(
		    "Usage: exemples_PIR_gpu\n"
		    "       exemples_PIR_gpu row column\n");
		exit(1);
	}

	double sigma8  = ldexp(1.0, 2 - 8 * KAPPABASE);
	double sigma5  = ldexp(1.0, 2 - 5 * KAPPABASE);
	double sigma6  = ldexp(1.0, 2 - 6 * KAPPABASE);
	MODULE* module = pvda_new_module_info(NBASE);

	gpu_ntt_initialize(NBASE);

	// Automorphims keys
	double ksk_sigma = sigma8;
	GLWEParams* params_glwe_autokey =
	    new_glwe_params(NBASE, KBASE, KAPPABASE, 16, ksk_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* auto_ksk_params = new_glwegadget_params(params_glwe_autokey, KAPPABASE, 8);

	//GGSW keys
	double ggsw_ksk_sigma = sigma8;
	GLWEParams* params_ggsw_change_key =
	    new_glwe_params(NBASE, KBASE, KAPPABASE, 16, ggsw_ksk_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* auto_ggsw_params = new_ggsw_params(params_glwe_autokey, KBASE, KAPPABASE, 16);

	//Input queries
	double row_sigma             = sigma8;
	GLWEParams* params_row_query = new_glwe_params(NBASE, KBASE, KAPPABASE, 16, row_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* row_query_gad_params = new_glwegadget_params(params_row_query, KAPPABASE, L_TILDE_Q1);
	double col_sigma             = sigma8;
	GLWEParams* params_col_query = new_glwe_params(NBASE, KBASE, KAPPABASE, 16, col_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* col_query_gad_params = new_glwegadget_params(params_col_query, KAPPABASE, 8);

	//Expanded queries (parameters for the unpacked form)
	double row_exp_sigma = sigma6;
	GLWEParams* row_exp_params =
	    new_glwe_params(NBASE, KBASE, KAPPABASE, 12, row_exp_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* row_exp_gad_params = new_glwegadget_params(row_exp_params, KAPPABASE, L_TILDE_Q1);

	double col_exp_sigma = sigma6;
	GLWEParams* col_exp_params =
	    new_glwe_params(NBASE, KBASE, KAPPABASE, 12, col_exp_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* col_exp_ggsw_params = new_ggsw_params(col_exp_params, KBASE, KAPPABASE, 10);

	double col_sum_sigma = sigma6;
	GLWEParams* col_sum_params =
	    new_glwe_params(NBASE, KBASE, KAPPABASE, 12, col_sum_sigma, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEParams* final_params = new_glwe_params(NBASE, KBASE, KAPPABASE, 10, sigma5, NOISE_UNIFORM_POWER_OF_TWO);

	GLWEParams* db_params = new_glwe_params(NBASE, KBASE, KAPPABASE, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);

	//Client phase 0: secret and evaluation key generation
	GLWESecretKeyPrepared* sk_prep;
	GLWEAutomorphismKSKCollection* ksks;
	GGSWCiphertextPrep** ggsw_ksks;

	onionpir_client_phase0(module, &sk_prep, 2, final_params, &ksks, auto_ksk_params, &ggsw_ksks, auto_ggsw_params);

	// Server pre-processing: generate the database columns and upload them to
	// the GPU once, ahead of any query.
	GLWECiphertext* res = new_glwe(final_params);
	for (int c = 0; c <= IN_MEMORY_DFT_COLS; ++c)
	{
		prepare_column_gpu(module, c, db_params, row_exp_gad_params);
	}

	GLWECiphertext* row_query;
	GLWECiphertext* col_query;
	while (1)
	{
		int row, col;

		if (argc == 3)
		{
			row = atoi(argv[1]);
			col = atoi(argv[2]);
		}
		else
		{
			printf("Input the row and column to select, space-separated: ");
			int st = scanf("%d %d", &row, &col);
			if (st != 2)
			{
				printf("Wrong selection format\n");
				exit(1);
			}
		}

		printf("Selecting row %d, column %d\n", row, col);
		--row;
		--col;
		onionpir_client_phase1(module, &row_query, &col_query, sk_prep, row, col, params_row_query, params_col_query,
		                       row_query_gad_params, col_query_gad_params);

		struct timespec server_start;
		clock_gettime(CLOCK_REALTIME, &server_start);

		onionpir_server_gpu(module, auto_ggsw_params, row_exp_gad_params, db_params, col_sum_params, ksks,
		                    (const GGSWCiphertextPrep**)ggsw_ksks, res, row_query, col_query);

		struct timespec server_end;
		clock_gettime(CLOCK_REALTIME, &server_end);

		double ms_elapsed =
		    (server_end.tv_sec - server_start.tv_sec) * 1000 + (server_end.tv_nsec - server_start.tv_nsec) / 1000000;
		printf("Server elapsed time: %.2f ms\n", ms_elapsed);

		printf("Result:\n");
		print_coefs_glwe(module, res, sk_prep, 4, SHFT_AMT);

		double throughput_bits_sec = (64.0 - SHFT_AMT) * NBASE * MATRIX_ROWS * MATRIX_COLS * 1000 / ms_elapsed;
		printf("Server throughput: %.2f MiB/s\n", throughput_bits_sec / 8 / 1024 / 1024);
		fflush(stdout);

		delete_glwe(row_query);
		row_query = NULL;
		delete_glwe(col_query);
		col_query = NULL;

		if (argc == 3) break;
		printf("\n");
	}

	delete_glwe(res);
	delete_automorphism_ksk_collection(ksks, 1);
	for (int i = 0; i < KBASE; ++i)
	{
		delete_ggsw_prep(ggsw_ksks[i]);
	}
	free(ggsw_ksks);

	delete_glwe_secret_key_prepared(sk_prep);

	for (int c = 0; c <= IN_MEMORY_DFT_COLS; ++c)
		if (columns_gpu[c]) pvda_gpu_free(columns_gpu[c]);

	pvda_delete_module_info(module);

	delete_glwe_params(params_glwe_autokey);
	delete_glwegadget_params(auto_ksk_params);

	delete_glwe_params(params_ggsw_change_key);
	delete_ggsw_params(auto_ggsw_params);

	delete_glwe_params(params_row_query);
	delete_glwegadget_params(row_query_gad_params);
	delete_glwe_params(params_col_query);
	delete_glwegadget_params(col_query_gad_params);

	delete_glwe_params(row_exp_params);
	delete_glwegadget_params(row_exp_gad_params);
	delete_glwe_params(col_exp_params);
	delete_ggsw_params(col_exp_ggsw_params);

	delete_glwe_params(col_sum_params);
	delete_glwe_params(final_params);
	delete_glwe_params(db_params);

	return 0;
}
