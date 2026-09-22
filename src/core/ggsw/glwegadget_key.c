#include "glwegadget_key.h"

#include <stdint.h>
#include <stdlib.h>

#include "ggsw_params.h"
#include "glwegadget_ciphertext.h"
#include "utils.h"

static inline int allocate_prepared_gadget_array(GLWEGadgetParams* params, GLWEGadgetCiphertextPrep** array, int n)
{
	int i = 0;
	for (int i = 0; i < n; ++i)
	{
		array[i] = new_glwegadget_prep(params);
		CHECK_ALLOC(array[i], "KSK allocation failed");
	}
	return 0;
cleanup:
	for (int i = 0; i < n; ++i)
	{
		delete_glwegadget_prep(array[i]);
	}
	return -1;
}

GLWEKSK* new_glwe_ksk(GLWEGadgetParams* params)
{
	GLWEKSK* ksk = malloc(sizeof(GLWEAutomorphismKey));
	CHECK_ALLOC(ksk, "Automorphism key allocation failed");
	ksk->params = params;

	ksk->enc_s = (GLWEGadgetCiphertextPrep**)calloc(params->params_glwe->k, sizeof(GLWEGadgetCiphertextPrep*));
	CHECK_ALLOC(ksk, "Automorphism key allocation failed");
	CHECK_CALL(allocate_prepared_gadget_array(params, ksk->enc_s, params->params_glwe->k),
	           "Failed to allocate automorphism key");

	return ksk;
cleanup:
	if (ksk) free((void*)ksk->enc_s);
	free(ksk);
	return NULL;
}

void delete_glwe_ksk(GLWEKSK* glwe_ksk)
{
	if (!glwe_ksk) return;
	for (int i = 0; i < glwe_ksk->params->params_glwe->k; ++i)
	{
		delete_glwegadget_prep(glwe_ksk->enc_s[i]);
	}
	free(glwe_ksk->enc_s);
	free(glwe_ksk);
}

GLWEAutomorphismKey* new_automorphism_key(GLWEGadgetParams* params)
{
	GLWEAutomorphismKey* auto_key = malloc(sizeof(GLWEAutomorphismKey));
	CHECK_ALLOC(auto_key, "Automorphism key allocation failed");

	auto_key->automorphism_p = -2;

	auto_key->glwe_ksk = new_glwe_ksk(params);
	CHECK_ALLOC(auto_key->glwe_ksk, "Automorphism key allocation failed");

	return auto_key;
cleanup:
	if (auto_key) delete_glwe_ksk(auto_key->glwe_ksk);
	free(auto_key);
	return NULL;
}

GLWEAutomorphismKey* new_automorphism_key_on(const PvdaBackend* backend, GLWEGadgetParams* params)
{
	GLWEAutomorphismKey* auto_key = new_automorphism_key(params);
	CHECK_ALLOC(auto_key, "Automorphism key allocation failed");
	if (pvda_backend_type(backend) == PVDA_BACKEND_SPQLIOS) return auto_key;

	// Place the prepared gadgets on the backend: compute_automorphism_key then prepares them there
	for (uint64_t i = 0; i < params->params_glwe->k; ++i)
	{
		GLWEGadgetCiphertextPrep* gadget = auto_key->glwe_ksk->enc_s[i];
		pvda_free_any(gadget->mat);
		gadget->mat = pvda_alloc_on(backend, glwegadget_coef_number(params) * sizeof(MatBivDFT));
		CHECK_ALLOC(gadget->mat, "Automorphism key allocation failed");
	}
	return auto_key;
cleanup:
	delete_automorphism_key(auto_key);
	return NULL;
}

void delete_automorphism_key(GLWEAutomorphismKey* automorphism_key)
{
	if (!automorphism_key) return;
	delete_glwe_ksk(automorphism_key->glwe_ksk);
	free(automorphism_key);
}

GLWEAutomorphismKeyCollection* new_automorphism_key_collection(uint64_t size)
{
	GLWEAutomorphismKeyCollection* collection = malloc(sizeof(GLWEAutomorphismKeyCollection));
	CHECK_ALLOC(collection, "Failed allocation of automorphism key collection");
	collection->keys = calloc(size, sizeof(GLWEAutomorphismKey*));
	CHECK_ALLOC(collection->keys, "Failed allocation of automorphism key collection");
	collection->size = size;
	return collection;

cleanup:

	if (collection) free(collection->keys);
	free(collection);
	return NULL;
}

GLWEAutomorphismKey* glwegadget_key_collection_put_key(GLWEAutomorphismKeyCollection* collection,
                                                       GLWEAutomorphismKey* key, uint64_t pos)
{
	if (pos >= collection->size)
	{
		return (void*)-1;
	}
	GLWEAutomorphismKey* tmp = collection->keys[pos];
	collection->keys[pos]    = key;
	return tmp;
}

GLWEAutomorphismKey* glwegadget_key_collection_get_key(const GLWEAutomorphismKeyCollection* collection, uint64_t pos)
{
	if (pos >= collection->size) return NULL;
	return collection->keys[pos];
}

uint64_t delete_automorphism_key_collection(GLWEAutomorphismKeyCollection* automorphism_keys, int deallocate_keys)
{
	uint64_t count = 0;
	if (deallocate_keys)
	{
		for (uint64_t i = 0; i < automorphism_keys->size; ++i)
		{
			if (automorphism_keys->keys[i]) ++count;
			delete_automorphism_key(automorphism_keys->keys[i]);
		}
	}
	free(automorphism_keys->keys);
	free(automorphism_keys);
	return count;
}
