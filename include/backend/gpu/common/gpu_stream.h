#ifndef PVDA_GPU_STREAM_H
#define PVDA_GPU_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque CUDA stream handle (cudaStream_t), exposed as void* so this header
 * can be included from plain C translation units that never see
 * <cuda_runtime.h>. NULL always denotes the default stream. */

/* Create a new, non-default CUDA stream. Caller owns it: release with
 * pvda_gpu_stream_destroy once no longer selected as the active stream. */
void* pvda_gpu_stream_create(void);

/* Destroy a stream created by pvda_gpu_stream_create. */
void pvda_gpu_stream_destroy(void* stream);

/* Block the calling host thread until all work already queued on `stream`
 * completes. Pass NULL to synchronize the default stream. */
void pvda_gpu_stream_synchronize(void* stream);

/* Push `stream` as the thread-local active stream used by every GPU backend
 * kernel launch / memcpy issued from this host thread from this point on.
 * Pass NULL to explicitly select the default stream. Every push must be
 * paired with a pvda_gpu_stream_pop to restore the previous active stream —
 * pushes nest, so this composes across call sites. */
void pvda_gpu_stream_push(void* stream);

/* Restore the active stream to what it was before the matching push. */
void pvda_gpu_stream_pop(void);

/* Returns this thread's current active stream (NULL means default stream). */
void* pvda_gpu_stream_get_active(void);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

#include <cuda_runtime.h>

/* The stream every GPU backend kernel launch / memcpy uses, for this host
 * thread. Defaults to the default stream (0). Prefer GpuStreamGuard (C++) or
 * pvda_gpu_stream_push/pop (C) over assigning this directly, so nested scopes
 * compose and always restore the previous stream. */
extern thread_local cudaStream_t gpu_active_stream;

/* RAII scope guard: activates `stream` as gpu_active_stream for this thread
 * and restores the previous value on scope exit. Safe to nest. */
class GpuStreamGuard {
   public:
	explicit GpuStreamGuard(cudaStream_t stream) : previous_(gpu_active_stream) { gpu_active_stream = stream; }
	~GpuStreamGuard() { gpu_active_stream = previous_; }

	GpuStreamGuard(const GpuStreamGuard&)            = delete;
	GpuStreamGuard& operator=(const GpuStreamGuard&) = delete;

   private:
	cudaStream_t previous_;
};

#endif  // __CUDACC__

#endif  // PVDA_GPU_STREAM_H
