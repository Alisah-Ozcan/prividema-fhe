# GPU (CUDA) backend {#gpu_backend}

Prividema-fhe can run its core operations on an NVIDIA GPU. The GPU support is compiled
only when the library is configured with `ENABLE_CUDA=ON`; a build without it contains no
CUDA code and does not depend on the CUDA toolkit.

## Building

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = Ada, e.g. RTX 4090
cmake --build build
```

`CMAKE_CUDA_ARCHITECTURES` defaults to a list covering Maxwell to Hopper; setting it to the
target GPU only shortens the build. The GPU NTT library
([GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)) is fetched automatically.

## Using the CUDA backend

The backend decides where objects live and where operations run. Objects created with the
`*_on` constructors on a CUDA backend live in GPU memory, and every operation applied to them
runs on the GPU. The same code runs on the CPU when given a spqlios backend:

```c
PvdaBackend* backend = pvda_new_cuda_backend(4096);   // or pvda_new_spqlios_backend(4096)
if (!backend) { /* built without ENABLE_CUDA, or no CUDA device */ }

GLWECiphertext* ct   = glwe_clone_on(backend, host_ct);          // host -> GPU
GGSWCiphertext* ggsw = ggsw_clone_on(backend, host_ggsw);
GGSWCiphertextPrep* ggsw_prep = new_ggsw_prep_on(backend, ggsw_params);
ggsw_prepare(backend, ggsw_prep, ggsw);                           // prepared on the GPU

GLWECiphertext* res = new_glwe_on(backend, params);
ggsw_external_product(backend, res, ct, ggsw_prep);               // runs on the GPU
glwe_copy_to_host(host_res, res);                                 // GPU -> host

delete_glwe(res);                                                 // regular delete_* functions
```

`pvda_new_backend` selects the backend from a configuration:

```c
PvdaBackendConfig config = {.nn = 4096, .type = PVDA_BACKEND_CUDA};
PvdaBackend* backend     = pvda_new_backend(config);
```

### Constructors

| Host only (unchanged) | Placed on a backend |
|---|---|
| `new_glwe` | `new_glwe_on`, `glwe_clone_on` |
| `new_glwe_dft` | `new_glwe_dft_on` |
| `new_ggsw` / `new_ggsw_prep` | `new_ggsw_on`, `ggsw_clone_on` / `new_ggsw_prep_on` |
| `new_glwegadget` / `new_glwegadget_prep` | `new_glwegadget_on`, `glwegadget_clone_on` / `new_glwegadget_prep_on` |
| `new_automorphism_key` | `new_automorphism_key_on` |

`glwe_copy_to_host` brings a ciphertext back from either side. Objects from both families are
freed with the usual `delete_*` functions.

### What runs on the GPU

GLWE addition, subtraction, normalization and copy, `glwe_dft_to_coef`, the GGSW external
products (unprepared, prepared, to DFT), GGSW and GLWEGadget preparation, the GLWEGadget
half-products, `glwegadget_automorphism`, `glwe_trace_expand` (and the packed trace
expansions built on it), `generate_glwegad_to_ggsw_ksk` and the HEBits CMux and CMux tree.

Operations without a GPU implementation, in particular encryption, decryption and key
generation (they need the random number generator), run on the CPU on host objects. The CUDA
backend embeds a spqlios backend for them, so they can be called with the CUDA backend too;
the results are then moved to the GPU with the `*_clone_on` functions.

### Rules

- **Use a CUDA backend with GPU objects.** In a CUDA build, a spqlios backend given GPU objects
  logs an error and returns -1 (it recognizes the GPU memory allocated by the library, without
  using the CUDA runtime: host-only programs never initialize CUDA).
- **Keep the operands of an operation on one side.** An operation whose operands mix host and
  GPU memory logs an error and returns -1 instead of running (void functions leave their
  result untouched). The half-products accept their `a` operand on either side.
- **Prepare keys for the backend that uses them.** Prepared objects are backend-specific: the
  CPU uses a spqlios FFT (floating point) representation and the GPU an NTT (64-bit modular)
  one, which have the same size but are not interchangeable. Create prepared GGSWs, prepared
  GLWEGadgets and automorphism keys with the `*_on` constructors of the backend that will use
  them, and prepare them from raw objects on the same backend.
- The same holds for DFT-domain and prepared vectors (`pvda_vec_znx_dft`,
  `pvda_vmp_prepare_vec` / `biv_coefs_to_prep`): the GPU DFT domain is an NTT domain, so
  compute them on the backend that consumes them. The OnionPIR GPU example prepares its
  database columns on the GPU once, as the CPU example does on the CPU.

## How it works

The core (GLWE, GLWEGadget, GGSW) and scheme code contains no GPU code: its algorithms are
written once against the backend's virtual table of polynomial operations (`pvda_vec_znx_add`,
`pvda_vmp_apply_dft`, `pvda_vec_znx_automorphism`, ...). The CUDA backend implements that table
for GPU operands (`src/backend/impls/cuda_alias.c`, `src/backend/gpu/host/vtable_ops_host.cu`)
and forwards host operands to its embedded spqlios table. The GPU DFT domain is the NTT domain
modulo a 64-bit prime.

Temporaries of an operation are allocated next to its operands (`pvda_alloc_like`,
`new_glwe_like`), and objects are allocated, freed and copied through the backend's memory
functions (`pvda_alloc_on`, `pvda_free_any`, `pvda_memcpy_any`), so the same core code runs on
host and GPU data. Preparing a GGSW or GLWEGadget whose raw or prepared matrix is in GPU memory
prepares it on the GPU (`pvda_vmp_prepare_contiguous_placed`).

GPU operations are asynchronous: they are queued on the active stream and results become
visible to the host through the downloading functions (`glwe_copy_to_host`, ...), which wait
for them. Call `pvda_gpu_device_synchronize()` before reading a wall-clock timer around GPU
work. Independent work can be spread over streams with `pvda_gpu_stream_push` /
`pvda_gpu_stream_pop` (see `examples/example_PIR_gpu_stream.c`).

## Errors

A failing GPU operation (out of memory, a CUDA error, ...) never aborts the program. The
failure is recorded and the operation reports it the way the rest of the library does: an
allocation returns NULL and an operation of the backend returns -1, with a message in the log.

```c
GLWECiphertext* ct = new_glwe_on(backend, params);
if (!ct) { /* out of GPU memory */ }
if (ggsw_external_product(backend, res, ct, sel) != 0) { /* see pvda_gpu_last_error() */ }
```

`pvda_gpu_take_error()` tells whether a GPU failure happened since the last call (and clears
the flag), `pvda_gpu_last_error()` returns its message. Both are per-thread.

## GPU memory

GPU buffers come from a stream-ordered memory pool (`cudaMallocAsync`). Freed memory stays
cached for reuse, which keeps the many short-lived buffers of the GPU operations cheap. The
cache is unbounded by default; when an allocation runs out of memory the cache is emptied and
the allocation retried.

- `PVDA_GPU_POOL_CACHE_MB=<MiB>` bounds the cache, useful when several processes share the
  GPU (e.g. running the unit tests in parallel).
- `pvda_gpu_release_cached_memory()` returns all cached, unused memory to the driver.

## Examples

`examples/example_PIR_gpu.c` is the OnionPIR server (see the OnionPIR page) running on the CUDA
backend; `examples/example_PIR_gpu_stream.c` additionally spreads the column half-products
over two CUDA streams. Both are built when `ENABLE_CUDA=ON` and `BUILD_EXAMPLES=ON`:

```bash
CUDA_MODULE_LOADING=EAGER ./build/examples/example_PIR_gpu 3 5
```

`CUDA_MODULE_LOADING=EAGER` loads the GPU kernels at start-up instead of on first use, which
keeps the first query from paying that cost.
