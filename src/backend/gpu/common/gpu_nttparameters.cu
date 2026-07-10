#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "gpu/common/gpu_nttparameters.h"

static int bitreverse(int index, int n_power)
{
	int res = 0;
	for (int i = 0; i < n_power; i++)
	{
		res <<= 1;
		res = (index & 1) | res;
		index >>= 1;
	}
	return res;
}

static std::vector<Root64> generate_ntt_table(std::vector<Data64> psi, std::vector<Modulus64> primes, int n_power)
{
	int n_ = 1 << n_power;
	std::vector<Root64> forward_table;
	for (size_t i = 0; i < primes.size(); i++)
	{
		std::vector<Root64> table;
		table.push_back(1);
		for (int j = 1; j < n_; j++) table.push_back(OPERATOR64::mult(table[j - 1], psi[i], primes[i]));
		for (int j = 0; j < n_; j++) forward_table.push_back(table[bitreverse(j, n_power)]);
	}
	return forward_table;
}

static std::vector<Root64> generate_intt_table(std::vector<Data64> psi, std::vector<Modulus64> primes, int n_power)
{
	int n_ = 1 << n_power;
	std::vector<Root64> inverse_table;
	for (size_t i = 0; i < primes.size(); i++)
	{
		std::vector<Root64> table;
		table.push_back(1);
		Data64 inv_root = OPERATOR64::modinv(psi[i], primes[i]);
		for (int j = 1; j < n_; j++) table.push_back(OPERATOR64::mult(table[j - 1], inv_root, primes[i]));
		for (int j = 0; j < n_; j++) inverse_table.push_back(table[bitreverse(j, n_power)]);
	}
	return inverse_table;
}

NTTParameterGenerator& NTTParameterGenerator::instance()
{
	static NTTParameterGenerator inst;
	return inst;
}

NTTParameterGenerator::NTTParameterGenerator() {}

void NTTParameterGenerator::initialize(size_t size)
{
	std::lock_guard<std::mutex> lk(mtx_);
	int n_power = int(log2l(size));

	if (n_power > 20) throw std::runtime_error("Root of unity is not valid for the ringsize!");

	if (cache_.find(size) != cache_.end()) return;

	auto current_psi = OPERATOR64::exp(root_of_unity_, static_cast<Data64>(1 << (20 - n_power)), carrier_prime_);

	std::vector<Root64> cpu_ntt  = generate_ntt_table({current_psi}, {carrier_prime_}, n_power);
	std::vector<Root64> cpu_intt = generate_intt_table({current_psi}, {carrier_prime_}, n_power);

	VEC_GPU<Root64> ntt_buf(cpu_ntt.size());
	VEC_GPU<Root64> intt_buf(cpu_intt.size());
	ntt_buf.copy_from_host(cpu_ntt.data(), cpu_ntt.size());
	intt_buf.copy_from_host(cpu_intt.data(), cpu_intt.size());
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));

	Ninverse64 n_inv = OPERATOR64::modinv(static_cast<Data64>(size), carrier_prime_);

	cache_.emplace(size, NTTInfo{std::move(ntt_buf), std::move(intt_buf), n_inv});
}

Root64* NTTParameterGenerator::get_ntt_table(size_t size)
{
	initialize(size);
	return cache_.at(size).ntt_table_.data();
}

Root64* NTTParameterGenerator::get_intt_table(size_t size)
{
	initialize(size);
	return cache_.at(size).intt_table_.data();
}

Ninverse64 NTTParameterGenerator::get_n_inv(size_t size)
{
	initialize(size);
	return cache_.at(size).n_inv_;
}

Modulus64 NTTParameterGenerator::get_modulus() { return carrier_prime_; }

extern "C" {

void gpu_ntt_initialize(size_t n) { NTTParameterGenerator::instance().initialize(n); }

const uint64_t* gpu_ntt_get_table(size_t n)
{
	return reinterpret_cast<const uint64_t*>(NTTParameterGenerator::instance().get_ntt_table(n));
}

const uint64_t* gpu_ntt_get_intt_table(size_t n)
{
	return reinterpret_cast<const uint64_t*>(NTTParameterGenerator::instance().get_intt_table(n));
}

uint64_t gpu_ntt_get_modulus(void) { return NTTParameterGenerator::instance().get_modulus().value; }

uint64_t gpu_ntt_get_n_inv(size_t n) { return NTTParameterGenerator::instance().get_n_inv(n); }

}  // extern "C"
