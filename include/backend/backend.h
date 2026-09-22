#ifndef PVDA_BACKEND_H
#define PVDA_BACKEND_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Opaque internal type for the prividema backend object type
 */
typedef struct pvda_module_t PvdaBackend;

/**
 * @brief Kinds of Prividema-FHE backends
 */
typedef enum pvda_backend_type_t {
	PVDA_BACKEND_SPQLIOS = 0,  ///< CPU backend (spqlios-arithmetic, AVX)
	PVDA_BACKEND_CUDA    = 1,  ///< GPU backend (CUDA), only available when built with ENABLE_CUDA
} PvdaBackendType;

/**
 * @brief Configuration struct for a Prividema-FHE backend
 *
 * Still very much a workd in progress
 */
typedef struct pvda_module_config_t
{
	uint64_t nn;
	PvdaBackendType type;  ///< Defaults to PVDA_BACKEND_SPQLIOS when zero-initialized

} PvdaBackendConfig;

//Legacy backend port functions
//
/**
 * @brief Utility function (and stopgap function until
 * we have the config object system working) to create a
 * backend that uses the spqlios backend
 *
 * @param nn Parameter N
 */
PvdaBackend* pvda_new_spqlios_backend(int nn);

/**
 * @brief Creates a backend that runs on the GPU with CUDA
 *
 * Objects created on this backend (see the *_on constructors) live in GPU memory and
 * operations on them run on the GPU. Host-side work that has no GPU implementation
 * (encryption, key generation, decryption) is done on the CPU by the backend's embedded
 * spqlios backend, on host objects. Creating the backend prepares the GPU for parameter
 * N (NTT tables, memory pool).
 *
 * @param nn Parameter N
 *
 * @return NULL if the library was built without ENABLE_CUDA or no CUDA device is
 *         available, a pointer to the new backend otherwise
 */
PvdaBackend* pvda_new_cuda_backend(int nn);

/**
 * @brief Create a new backend according to the configuration struct
 *
 * @param config A backend configuration struct
 *
 * @return NULL in case of failure, a pointer to the new backend otherwise
 */
PvdaBackend* pvda_new_backend(PvdaBackendConfig config);

/**
 * @brief Retrieves the kind of a Prividema-FHE backend
 *
 * @param backend The backend
 *
 * @return The backend type
 */
PvdaBackendType pvda_backend_type(const PvdaBackend* backend);

/**
 * @brief Tells whether a buffer is in GPU memory, as seen by a backend
 *
 * @return 1 for GPU memory on a CUDA backend, 0 otherwise (always 0 on a spqlios backend)
 */
int pvda_on_gpu(const PvdaBackend* backend, const void* ptr);

/**
 * @brief Allocates a buffer on the same side (host or GPU memory) as another buffer
 *
 * Used for the temporaries of an operation, which must live where its operands live. On a
 * spqlios backend the buffer is always host memory (64-byte aligned).
 *
 * @param backend The backend
 * @param like    A buffer whose side the new buffer is placed on
 * @param bytes   The size of the new buffer
 *
 * @return NULL in case of failure, the new (uninitialized) buffer otherwise. Free with pvda_free_like
 */
void* pvda_alloc_like(const PvdaBackend* backend, const void* like, size_t bytes);

/**
 * @brief Frees a buffer from pvda_alloc_like (or any host malloc'd buffer)
 */
void pvda_free_like(const PvdaBackend* backend, void* ptr);

/**
 * @brief Delete a Prividema-FHE backend
 *
 * @param backend The backend to delete
 */
void pvda_delete_backend(PvdaBackend* backend);

/**
 * @brief Retrieves parameter N for a given Prividema-FHE backend
 *
 * @param backend The backend
 *
 * @return Parameter N
 */
uint64_t pvda_module_extract_nn(const PvdaBackend* backend);

#endif
