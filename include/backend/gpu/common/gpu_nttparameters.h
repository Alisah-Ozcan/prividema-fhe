#ifndef PVDA_GPU_NTTPARAMETERS_H
#define PVDA_GPU_NTTPARAMETERS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_ntt_initialize(size_t n);
const uint64_t* gpu_ntt_get_table(size_t n);
const uint64_t* gpu_ntt_get_intt_table(size_t n);
uint64_t gpu_ntt_get_modulus(void);
uint64_t gpu_ntt_get_n_inv(size_t n);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

#include <mutex>
#include <unordered_map>

#include "gpu/common/gpu_common.h"
#include "gpuntt/common/modular_arith.cuh"

class NTTParameterGenerator {
   public:
	static NTTParameterGenerator& instance();

	void initialize(size_t size);

	Root64* get_ntt_table(size_t size);
	Root64* get_intt_table(size_t size);
	Ninverse64 get_n_inv(size_t size);
	Modulus64 get_modulus();

   private:
	NTTParameterGenerator();
	~NTTParameterGenerator() = default;

	NTTParameterGenerator(const NTTParameterGenerator&)            = delete;
	NTTParameterGenerator& operator=(const NTTParameterGenerator&) = delete;

	struct NTTInfo
	{
		VEC_GPU<Root64> ntt_table_;
		VEC_GPU<Root64> intt_table_;
		Ninverse64 n_inv_;
	};

	const Modulus64 carrier_prime_ = Modulus64(1152921504634109953ULL);
	const Data64 root_of_unity_    = 1341959173301ULL;
	std::unordered_map<size_t, NTTInfo> cache_;
	std::mutex mtx_;
};

#endif  // __CUDACC__

#endif  // PVDA_GPU_NTTPARAMETERS_H
