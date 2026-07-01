#include <cuda_runtime.h>

#include "gpu/common/gpu_nttparameters.h"

extern "C" {

static int bitreverse(int index, int n_power)
{
	int res_1 = 0;
	for (int i = 0; i < n_power; i++)
	{
		res_1 <<= 1;
		res_1 = (index & 1) | res_1;
		index >>= 1;
	}
	return res_1;
}

static std::vector<Root64> generate_ntt_table(std::vector<Data64> psi, std::vector<Modulus64> primes, int n_power)
{
	int n_ = 1 << n_power;
	std::vector<Root64> forward_table;  // bit reverse order
	for (int i = 0; i < primes.size(); i++)
	{
		std::vector<Root64> table;
		table.push_back(1);

		for (int j = 1; j < n_; j++)
		{
			Data64 exp = OPERATOR64::mult(table[(j - 1)], psi[i], primes[i]);
			table.push_back(exp);
		}

		for (int j = 0; j < n_; j++)
		{
			forward_table.push_back(table[bitreverse(j, n_power)]);
		}
	}

	return forward_table;
}

static std::vector<Root64> generate_intt_table(std::vector<Data64> psi, std::vector<Modulus64> primes, int n_power)
{
	int n_ = 1 << n_power;
	std::vector<Root64> inverse_table;  // bit reverse order
	for (int i = 0; i < primes.size(); i++)
	{
		std::vector<Root64> table;
		table.push_back(1);

		Data64 inv_root = OPERATOR64::modinv(psi[i], primes[i]);
		for (int j = 1; j < n_; j++)
		{
			Data64 exp = OPERATOR64::mult(table[(j - 1)], inv_root, primes[i]);
			table.push_back(exp);
		}

		for (int j = 0; j < n_; j++)  // take bit reverse order
		{
			inverse_table.push_back(table[bitreverse(j, n_power)]);
		}
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

	if (n_power > 20)
	{
		throw std::runtime_error("Root of unity is not valid for the ringsize!");
	}

	auto it = cache_.find(size);
	if (it == cache_.end())
	{
		auto current_root_of_unity_ =
		    OPERATOR64::exp(root_of_unity_, static_cast<Data64>(1 << (20 - n_power)), carrier_prime_);

		// Forward NTT tables (bit reverse order)
		std::vector<Root64> cpu_ntt_table = generate_ntt_table({current_root_of_unity_}, {carrier_prime_}, n_power);
		VEC_GPU<Root64> ntt_table_in(cpu_ntt_table.size());
		ntt_table_in.copy_from_host(cpu_ntt_table.data(), cpu_ntt_table.size());

		// Inverse NTT tables (bit reverse order)
		std::vector<Root64> cpu_intt_table = generate_intt_table({current_root_of_unity_}, {carrier_prime_}, n_power);
		VEC_GPU<Root64> intt_table_in(cpu_intt_table.size());
		intt_table_in.copy_from_host(cpu_intt_table.data(), cpu_intt_table.size());
		cudaDeviceSynchronize();

		it = cache_.emplace(size, RootTables{std::move(ntt_table_in), std::move(intt_table_in)}).first;
	}
}

Root64* NTTParameterGenerator::get_ntt_table(size_t size)
{
	this->initialize(size);
	auto it = cache_.find(size);
	return it->second.ntt_table_.data();
}

Root64* NTTParameterGenerator::get_intt_table(size_t size)
{
	this->initialize(size);
	auto it = cache_.find(size);
	return it->second.intt_table_.data();
}

Modulus64 NTTParameterGenerator::get_modulus() { return carrier_prime_; }

// C wrapper for NTTParameterGenerator singleton
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

}  // extern "C"
