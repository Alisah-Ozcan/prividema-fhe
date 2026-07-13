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
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/common/gpu_stream.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "maths_structures.h"
#include "schemes/tfhe.h"
#include "spqlios_alias.h"
#include "univariate_polynomial.h"
#include "utils.h"

/*****************************************************************************
 * OnionPIR example using Half-products — GPU server.
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

// Number of CUDA streams the (single) server thread round-robins across
// while computing the 256 per-column half-products in onionpir_server_gpu.
// Column c runs on streams[c % NUM_STREAMS].
#define NUM_STREAMS 2

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
	// CMux-tree leaves — device-resident. A single contiguous allocation
	// sliced into MATRIX_COLS views, instead of MATRIX_COLS separate
	// pvda_new_glwe_device calls: that helper does its own cudaMalloc *and*
	// a host malloc(sizeof(GLWECiphertext)) per call, so 256 individual
	// calls means 256 host heap allocations and 256 device allocations
	// (each a driver-level sync point) for what is really one block of GPU
	// memory. Here the leaf structs themselves live on the stack (no host
	// heap allocation at all) and only their .vec pointers are sliced out
	// of the one device buffer, freed in a single pvda_gpu_free below.
	uint64_t leaf_elems          = glwe_coef_number(aggregation_params);
	int64_t* d_glwe_tree_storage = pvda_gpu_alloc((size_t)MATRIX_COLS * leaf_elems);
	GLWECiphertext glwe_tree_storage[MATRIX_COLS];
	GLWECiphertext* glwe_tree_first_level[MATRIX_COLS];
	for (int c = 0; c < MATRIX_COLS; ++c)
	{
		glwe_tree_storage[c].params = aggregation_params;
		glwe_tree_storage[c].vec    = (VecBiv*)(d_glwe_tree_storage + (size_t)c * leaf_elems);
		glwe_tree_first_level[c]    = &glwe_tree_storage[c];
	}

	GLWEGadgetParams* mega_params = new_glwegadget_params(query1_params->params_glwe, query1_params->kappa_tilde,
	                                                      query1_params->l_tilde * MATRIX_ROWS);

	// row_query/col_query are still per-query, CPU-encrypted ciphertexts (RNG
	// sampling has no GPU path) — upload them once so every downstream op
	// (glwegadget_automorphism, glwe_rotate_flattened_inplace, add_glwe, ...)
	// sees a device-resident source and takes its GPU branch.
	int64_t* d_row_query_vec     = pvda_glwe_to_device(row_query);
	GLWECiphertext row_query_dev = {.params = row_query->params, .vec = (VecBiv*)d_row_query_vec};
	int64_t* d_col_query_vec     = pvda_glwe_to_device(col_query);
	GLWECiphertext col_query_dev = {.params = col_query->params, .vec = (VecBiv*)d_col_query_vec};

	// --- Row query expansion — now fully on the GPU: row_trace_unprep is a
	// device buffer, packed_glwegadget_trace_expand's automorphism/rotation
	// steps dispatch through glwegadget_automorphism/glwe_trace_expand's GPU
	// branches (see gpu/host/vec_znx_rotate_automorphism_host.h), and
	// glwegadget_prepare below already dispatches to glwegadget_prepare_gpu
	// once its raw input is device-resident. ---
	GLWEGadgetCiphertext gadgets[MATRIX_ROWS];
	GLWEGadgetCiphertext* gptrs[MATRIX_ROWS];
	GLWEGadgetCiphertext* row_trace_unprep = pvda_new_glwegadget_device(mega_params);
	for (int r = 0; r < MATRIX_ROWS; ++r)
	{
		gadgets[r].params = query1_params;
		gadgets[r].mat    = glwegadget_extract_bivglwe(row_trace_unprep, 1 + r * query1_params->l_tilde);
		gptrs[r]          = &gadgets[r];
	}
	packed_glwegadget_trace_expand(module, gptrs, MATRIX_ROWS, L_TILDE_Q1, &row_query_dev, ksks);

	GLWEGadgetCiphertextPrep* glwegad_trace = new_glwegadget_prep(mega_params);
	glwegadget_prepare(module, glwegad_trace, row_trace_unprep);

	delete_glwegadget(row_trace_unprep);

	// --- Half products (the main performance bottleneck) — fully on the GPU.
	// Single thread, but round-robins gpu_active_stream across NUM_STREAMS
	// CUDA streams (column c on streams[c % NUM_STREAMS]) instead of running
	// every column on the implicit default stream. Each stream slot gets its
	// own scratch DFT buffer so consecutive columns on different slots never
	// share a buffer. ---
	struct timespec server_start;
	clock_gettime(CLOCK_REALTIME, &server_start);

	void* streams[NUM_STREAMS];
	GLWECiphertextDFT tmp_glwe_dft_gpu[NUM_STREAMS];
	for (int s = 0; s < NUM_STREAMS; ++s)
	{
		streams[s] = pvda_gpu_stream_create();
		tmp_glwe_dft_gpu[s] =
		    (GLWECiphertextDFT){.params = aggregation_params,
		                        .vec    = (VecBivDFT*)pvda_gpu_alloc((size_t)glwe_params_n_limbs(aggregation_params) *
		                                                             (size_t)aggregation_params->nn)};
	}

	for (int64_t c = 0; c < MATRIX_COLS; ++c)
	{
		int slot = (int)(c % NUM_STREAMS);

		pvda_gpu_stream_push(streams[slot]);
		glwegadget_half_prod_prepared_to_dft(module, &tmp_glwe_dft_gpu[slot], glwegad_trace,
		                                     (const PolyBivPrep*)onionpir_get_prepared_column_gpu(c));
		glwe_dft_to_coef(module, glwe_tree_first_level[c], &tmp_glwe_dft_gpu[slot]);
		pvda_gpu_stream_pop();
	}

	for (int s = 0; s < NUM_STREAMS; ++s)
	{
		pvda_gpu_free((int64_t*)tmp_glwe_dft_gpu[s].vec);
		pvda_gpu_stream_destroy(streams[s]);
	}

	struct timespec server_end;
	clock_gettime(CLOCK_REALTIME, &server_end);
	double ms_elapsed =
	    (server_end.tv_sec - server_start.tv_sec) * 1000 + (server_end.tv_nsec - server_start.tv_nsec) / 1000000;
	printf("HP (GPU, %d streams) elapsed time: %.2f ms\n", NUM_STREAMS, ms_elapsed);

	delete_glwegadget_prep(glwegad_trace);

	// --- Column-selection query expansion — also fully on the GPU now:
	// ggsw_trace_raw[] is device-resident, packed_glwegadget_trace_expand_ggsw's
	// internal glwe_trace_expand + ggsw_external_product calls dispatch to
	// their GPU branches, and ggsw_prepare below already dispatches to
	// ggsw_prepare_gpu once its raw input is device-resident. ---
	GGSWCiphertext* ggsw_trace_raw[LOG2_COLS] = {0};
	for (int r = 0; r < LOG2_COLS; ++r) ggsw_trace_raw[r] = pvda_new_ggsw_device(ggsw_ksk_params);

	packed_glwegadget_trace_expand_ggsw(module, ggsw_trace_raw, LOG2_COLS, ggsw_params_l_tilde_a(ggsw_ksk_params),
	                                    &col_query_dev, ksks, ggsw_ksks);

	GGSWCiphertextPrep* ggsw_trace[LOG2_COLS] = {0};
	for (int r = 0; r < LOG2_COLS; ++r)
	{
		ggsw_trace[r] = new_ggsw_prep(ggsw_ksk_params);
		ggsw_prepare(module, ggsw_trace[r], ggsw_trace_raw[r]);
		delete_ggsw(ggsw_trace_raw[r]);
	}

	pvda_gpu_free(d_row_query_vec);
	pvda_gpu_free(d_col_query_vec);

	// --- CMux selection tree — entirely on the GPU: leaves, selectors and the
	// result are all device-resident, and tfhe_cmux_tree allocates its own
	// intermediate nodes on the device too (see tfhe_cmux_tree_new_node).
	// delete_src=0: the leaves are slices of the single d_glwe_tree_storage
	// allocation (see above), not independent pvda_new_glwe_device buffers,
	// so tfhe_cmux_tree must not delete_glwe() them one by one (that would
	// cudaFree() an interior pointer and free() a stack address). We own
	// that buffer and release it in one pvda_gpu_free below instead. ---
	GLWECiphertext* res_gpu = pvda_new_glwe_device(res->params);
	tfhe_cmux_tree(module, res_gpu, (const GLWECiphertext**)glwe_tree_first_level, MATRIX_COLS,
	               (const GGSWCiphertextPrep**)ggsw_trace, LOG2_COLS, 0);

	pvda_glwe_from_device(res, res_gpu->vec);
	delete_glwe(res_gpu);

	// Single free for the whole CMux-leaf block (glwe_tree_storage[] itself
	// is stack memory — nothing to free there, only the device buffer it
	// slices into).
	pvda_gpu_free(d_glwe_tree_storage);

	for (int i = 0; i < LOG2_COLS; ++i) delete_ggsw_prep(ggsw_trace[i]);
	delete_glwegadget_params(mega_params);
	return 0;
}

// Swaps every enc_s[i] of a freshly-allocated (host-prepared-by-default) KSK
// for a device Prep placeholder, so the prepare_automorphism_key call right
// after this NTT-prepares it on the GPU (see glwegadget_arithmetic.c's
// #ifdef ENABLE_CUDA branch, gated on pvda_is_device_pointer(enc_s[i]->mat)).
// Must run BEFORE prepare_automorphism_key — CPU glwegadget_prepare produces
// a spqlios DFT-domain (double) matrix, GPU glwegadget_prepare_gpu an
// NTT-domain (int64 residue) one; same footprint, incompatible content, so
// there is no way to convert a CPU-prepared KSK into a GPU one after the
// fact other than re-running preparation from the raw ciphertext.
static void make_automorphism_ksk_device(GLWEAutomorphismKSK* ksk)
{
	uint64_t k = ksk->params->params_glwe->k;
	for (uint64_t i = 0; i < k; ++i)
	{
		delete_glwegadget_prep(ksk->enc_s[i]);
		ksk->enc_s[i] = pvda_new_glwegadget_prep_device(ksk->params);
	}
}

// Setup phase for the client in the protocol: secret and evaluation key
// generation. Same protocol as exemples_PIR.c's client phase0 — encryption
// itself is RNG-based and stays on the CPU — but the KSKs are NTT-prepared
// directly on the GPU (see make_automorphism_ksk_device above), a one-time
// setup cost, so every later per-query dispatch (glwegadget_automorphism,
// glwegadget_half_prod, ggsw_external_product) sees device-resident KSK
// material and takes its GPU branch.
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
		make_automorphism_ksk_device(ksk);
		prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	*ggsw_ksks_out = ggsw_ksks;
	for (int i = 0; i < KBASE; ++i) ggsw_ksks[i] = pvda_new_ggsw_prep_device(auto_ggsw_params);
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
		    "Usage: exemples_PIR_gpu_stream\n"
		    "       exemples_PIR_gpu_stream row column\n");
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
	double col_sigma                       = sigma8;
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
