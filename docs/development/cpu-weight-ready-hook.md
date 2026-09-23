# CPU weight readiness for streamed dense matrices

This fork provides two independent hooks. `ggml_cpu_set_expert_ready_hook` keeps its existing signature and per-expert `MUL_MAT_ID` call site. `ggml_cpu_set_weight_ready_hook` adds a readiness boundary for ordinary CPU `MUL_MAT`. Neither hook performs I/O or changes the numerical kernels.

The dense hook runs once on worker 0 before any worker enters the operation, including CPU extra-buffer and llamafile paths. It receives `src0`, which can be a view or an intermediate tensor. Return true when all its bytes are readable, or false to return `GGML_STATUS_ABORTED` without executing that operation or later graph nodes. Other workers wait at a barrier while the hook runs. With no hook registered, that barrier is absent.

```cpp
static bool weights_ready(const ggml_tensor * weight, void * user_data) {
    auto * stream = static_cast<WeightStream *>(user_data);
    // Unmanaged tensors are already readable. Wait only for managed weights.
    return !stream->serves(weight) || stream->wait(weight);
}

ggml_cpu_set_weight_ready_hook(weights_ready, &stream);
// Run CPU inference. Check the graph/decode result before using its outputs.
ggml_cpu_set_weight_ready_hook(nullptr, nullptr);
```

`WeightStream` above denotes the application's residency owner, not a class supplied by this fork. For dynamically loaded CPU backends, the registry exposes both setters by their public function names. The runnable example is `test-backend-ops --test-weight-ready`.

## Lifetime and concurrency

- Registration is process-wide, like the existing expert hook. Set or clear it only when all CPU graphs are idle. Keep callback data alive until every graph using it has returned. Independent concurrent sessions need one shared dispatcher with synchronized application state, or external serialization.
- A hook must not throw, start recursive compute, change graph structure, or rebind tensor pointers. Fill the existing backing storage. Views must already point at that storage.
- The producer must finish its writes before the hook returns true. Keep bytes resident through operation completion. Reclaim only after a known last consumer; a later hook is not a guarantee that a weight will never be used again.
- A blocking callback must implement its own cancellation/error wake-up. The graph's abort callback cannot interrupt a callback that never returns.
- This covers the complete `src0` of `MUL_MAT`, not row tiles, `src1`, gathers, norms, convolutions, or other operators. The application must keep those dependencies valid separately. `MUL_MAT_ID` continues to use the expert hook.

## Dense projection waves

An application can read upcoming gate/up/down matrices on a bounded worker while the CPU computes with complete current matrices. Before each matmul, the hook joins the required read. Read order, lane count, prefetch limits, memory accounting and eviction remain application policy. No model names, GGUF parsing, repacking, or storage settings enter this hook.

This hides some I/O latency; it does not reduce the compulsory weight bytes of a dense pass. It does not start a matmul against partially loaded rows. Keep native GGUF layouts and disable extra buffer types when application-owned storage requires that invariant. The new hook does not make repacked weights streamable.

## Verification

`ctest -R '^test-weight-ready$'` checks F32, F16, Q4_0, Q4_1, Q5_K, Q6_K and Q8_0 with one/four compute threads and 1/4/32 input tokens. A producer thread fills each matrix only after demand, then overwrites the preceding matrix after its consumer completes. Outputs must match the resident baseline byte for byte. Cases also cover offset views, first/second operation failures, abort reset, hook removal and coexistence with the existing expert hook.

Run the test with both native and OpenMP thread pools, and under sanitizers. Also run `test-backend-ops` for `MUL_MAT`/`MUL_MAT_ID` against the CPU reference implementation, the consuming runtime's MoE identity gates, and model-level checks. Performance depends on the application's residency policy and storage; the hook alone is not a dense streamer or a throughput claim.
