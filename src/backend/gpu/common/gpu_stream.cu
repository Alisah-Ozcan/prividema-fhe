#include <vector>

#include "gpu/common/gpu_common.h"
#include "gpu/common/gpu_stream.h"

thread_local cudaStream_t gpu_active_stream = 0;

namespace {
thread_local std::vector<cudaStream_t> gpu_stream_stack;
}

extern "C" {

void* pvda_gpu_stream_create(void)
{
	cudaStream_t stream = nullptr;
	CUDA_CHECK(cudaStreamCreate(&stream));
	return (void*)stream;
}

void pvda_gpu_stream_destroy(void* stream)
{
	if (!stream) return;
	CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
}

void pvda_gpu_stream_synchronize(void* stream) { CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream)); }

void pvda_gpu_stream_push(void* stream)
{
	gpu_stream_stack.push_back(gpu_active_stream);
	gpu_active_stream = (cudaStream_t)stream;
}

void pvda_gpu_stream_pop(void)
{
	if (gpu_stream_stack.empty())
	{
		fprintf(stderr, "pvda_gpu_stream_pop: stack underflow (pop without matching push)\n");
		abort();
	}
	gpu_active_stream = gpu_stream_stack.back();
	gpu_stream_stack.pop_back();
}

void* pvda_gpu_stream_get_active(void) { return (void*)gpu_active_stream; }

}  // extern "C"
