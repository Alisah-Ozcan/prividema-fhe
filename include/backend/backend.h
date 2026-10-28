#ifndef PVDA_BACKEND_H
#define PVDA_BACKEND_H

#include <stdint.h>

/**
 * @brief Opaque internal type for the prividema backend object type
 */
typedef struct pvda_module_t PvdaBackend;

/**
 * @brief Configuration struct for a Prividema-FHE backend
 *
 * Still very much a work in progress
 */
typedef struct pvda_module_config_t
{
	uint64_t nn;

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
 * @brief Create a new backend according to the configuration struct
 *
 * The backend is the one selected at compile time: a CUDA backend when the library is built
 * with ENABLE_CUDA=ON, a spqlios backend otherwise.
 *
 * @param config A backend configuration struct
 *
 * @return NULL in case of failure, a pointer to the new backend otherwise
 */
PvdaBackend* pvda_new_backend(PvdaBackendConfig config);

/**
 * @brief Delete a Prividema-FHE backend
 *
 * @param module The backend to delete
 */
void pvda_delete_backend(PvdaBackend* module);

/**
 * @brief Retrieves parameter N for a given Prividema-FHE backend
 *
 * @param module The backend
 *
 * @return Parameter N
 */
uint64_t pvda_module_extract_nn(const PvdaBackend* module);

#endif
