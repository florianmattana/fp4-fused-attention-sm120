#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <common.h>

// ── Benchmark configuration ───────────────────────────────────────────────────
static constexpr int N_WARMUP = 20;
static constexpr int N_RUNS   = 100;
static constexpr int BATCH    = 1;
static constexpr int HEADS    = 32;

// ── Timing helpers ────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end()  { cudaEventRecord(stop); cudaEventSynchronize(stop);
                   float ms; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// ── Single benchmark run ──────────────────────────────────────────────────────
template<int HEAD_DIM>
void benchmark(int seq_k)
{
    int seq_q   = BQ;
    int n_blocks = BATCH * HEADS;

    // Sizes
    size_t q_elems = (size_t)BATCH * HEADS * seq_q * HEAD_DIM;
    size_t k_elems = (size_t)BATCH * HEADS * seq_k * HEAD_DIM;

    // Allocate and fill with random data
    float *hQ = new float[q_elems];
    float *hK = new float[k_elems];
    float *hV = new float[k_elems];
    for (size_t i = 0; i < q_elems; i++) hQ[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (size_t i = 0; i < k_elems; i++) hK[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (size_t i = 0; i < k_elems; i++) hV[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ,   q_elems * sizeof(float));
    cudaMalloc(&dK,   k_elems * sizeof(float));
    cudaMalloc(&dV,   k_elems * sizeof(float));
    cudaMalloc(&dOut, q_elems * sizeof(float));

    // ── Measure H2D transfer ──────────────────────────────────────────────────
    GpuTimer t_h2d;
    t_h2d.begin();
    cudaMemcpy(dQ, hQ, q_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, k_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, k_elems * sizeof(float), cudaMemcpyHostToDevice);
    float ms_h2d = t_h2d.end();
    size_t bytes_in = (q_elems + 2 * k_elems) * sizeof(float);
    float bw_h2d = (bytes_in / 1e9f) / (ms_h2d * 1e-3f);

    // ── Warmup ────────────────────────────────────────────────────────────────
    for (int i = 0; i < N_WARMUP; i++)
        fused_fp4_attention<HEAD_DIM><<<n_blocks, NUM_THREADS>>>(
            dQ, dK, dOut, dV, seq_q, seq_k, HEADS);
    cudaDeviceSynchronize();

    // ── Kernel benchmark ──────────────────────────────────────────────────────
    GpuTimer t_kernel;
    t_kernel.begin();
    for (int i = 0; i < N_RUNS; i++)
        fused_fp4_attention<HEAD_DIM><<<n_blocks, NUM_THREADS>>>(
            dQ, dK, dOut, dV, seq_q, seq_k, HEADS);
    float ms_kernel = t_kernel.end() / N_RUNS;

    // ── Measure D2H transfer ──────────────────────────────────────────────────
    float *hOut = new float[q_elems];
    GpuTimer t_d2h;
    t_d2h.begin();
    cudaMemcpy(hOut, dOut, q_elems * sizeof(float), cudaMemcpyDeviceToHost);
    float ms_d2h = t_d2h.end();
    size_t bytes_out = q_elems * sizeof(float);
    float bw_d2h = (bytes_out / 1e9f) / (ms_d2h * 1e-3f);

    // ── Compute metrics ───────────────────────────────────────────────────────
    // FLOPs: 2 * batch * heads * seq_q * seq_k * head_dim  (Q×Kᵀ)
    //      + 2 * batch * heads * seq_q * seq_k * head_dim  (softmax×V)
    double flops    = 4.0 * BATCH * HEADS * seq_q * seq_k * HEAD_DIM;
    double tflops   = flops / (ms_kernel * 1e-3) / 1e12;

    // Bytes read/written by kernel (Q, K, V reads + Out write) — lower bound
    size_t bytes_kernel = (q_elems + 2 * k_elems + q_elems) * sizeof(float);
    float bw_kernel = (bytes_kernel / 1e9f) / (ms_kernel * 1e-3f);

    printf("| %8d | %5d | %9.3f | %8.3f | %10.2f | %10.2f | %10.2f |\n",
           HEAD_DIM, seq_k,
           ms_h2d + ms_kernel + ms_d2h,   // total latency
           ms_kernel,                       // kernel only
           tflops,
           bw_kernel,
           bw_h2d);

    delete[] hQ; delete[] hK; delete[] hV; delete[] hOut;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
}

int main()
{
    srand(42);

    printf("\nRTX 5070 Ti  SM120  FP4 fused attention\n");
    printf("batch=%d  heads=%d  seq_q=%d  warmup=%d  runs=%d\n\n",
           BATCH, HEADS, BQ, N_WARMUP, N_RUNS);

    printf("| %8s | %5s | %9s | %8s | %10s | %10s | %10s |\n",
           "head_dim", "seq_k", "total_ms", "kern_ms", "TFLOPS", "BW_kern GB/s", "BW_h2d GB/s");
    printf("|%s|\n", std::string(79, '-').c_str());

    benchmark<128>(128);
    benchmark<128>(256);
    benchmark<128>(512);
    benchmark<128>(1024);

    benchmark<64>(128);
    benchmark<64>(256);
    benchmark<64>(512);
    benchmark<64>(1024);

    return 0;
}