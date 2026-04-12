#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <common.h>

static constexpr int SEQ_Q = BQ;
static constexpr int HEAD  = Bd;
static constexpr int VCOL  = MMA_N;

void cpu_matmul(const float* A, const float* B, float* C,
                int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

void cpu_softmax_rows(float* S, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float* row = S + i * cols;
        float m = row[0];
        for (int j = 1; j < cols; j++) m = fmaxf(m, row[j]);
        float sum = 0.f;
        for (int j = 0; j < cols; j++) { row[j] = expf(row[j] - m); sum += row[j]; }
        for (int j = 0; j < cols; j++) row[j] /= sum;
    }
}

void cpu_attention(const float* Q, const float* K, const float* V,
                   float* Out, int seq_q, int seq_k, int head, int vcol)
{
    float* Kt = new float[seq_k * head];
    float* S  = new float[seq_q * seq_k];

    for (int i = 0; i < seq_k; i++)
        for (int j = 0; j < head; j++)
            Kt[j * seq_k + i] = K[i * head + j];

    cpu_matmul(Q, Kt, S, seq_q, seq_k, head);

    // 1/sqrt(d) scaling — must match GPU kernel
    float scale = 1.0f / sqrtf((float)head);
    for (int i = 0; i < seq_q * seq_k; i++) S[i] *= scale;

    cpu_softmax_rows(S, seq_q, seq_k);
    cpu_matmul(S, V, Out, seq_q, vcol, seq_k);

    delete[] Kt;
    delete[] S;
}

float cosine_similarity(const float* a, const float* b, int n)
{
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
}

void test_attention(int seq_k)
{
    printf("\n=== test fused_fp4_attention  seq_q=%d  seq_k=%d ===\n",
           SEQ_Q, seq_k);

    float* Q   = new float[SEQ_Q * HEAD];
    float* K   = new float[seq_k * HEAD];
    float* V   = new float[seq_k * VCOL];
    float* ref = new float[SEQ_Q * VCOL];
    float* out = new float[SEQ_Q * VCOL];

    srand(42);
    auto rnd = []{
        float vals[] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
        return vals[rand() % 5];
    };
    for (int i = 0; i < SEQ_Q * HEAD; i++) Q[i] = rnd();
    for (int i = 0; i < seq_k * HEAD; i++) K[i] = rnd();
    for (int i = 0; i < seq_k * VCOL; i++) V[i] = rnd();

    cpu_attention(Q, K, V, ref, SEQ_Q, seq_k, HEAD, VCOL);

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ,   SEQ_Q * HEAD * sizeof(float));
    cudaMalloc(&dK,   seq_k * HEAD * sizeof(float));
    cudaMalloc(&dV,   seq_k * VCOL * sizeof(float));
    cudaMalloc(&dOut, SEQ_Q * VCOL * sizeof(float));
    cudaMemcpy(dQ, Q, SEQ_Q * HEAD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, seq_k * HEAD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, seq_k * VCOL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, SEQ_Q * VCOL * sizeof(float));

    fused_fp4_attention<<<1, NUM_THREADS>>>(dQ, dK, dOut, dV, seq_k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("kernel launch error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaMemcpy(out, dOut, SEQ_Q * VCOL * sizeof(float), cudaMemcpyDeviceToHost);

    float cos_sim = cosine_similarity(ref, out, SEQ_Q * VCOL);
    printf("cosine similarity : %.4f\n", cos_sim);
    printf("%s (seuil 0.99)\n", cos_sim >= 0.99f ? "PASS" : "FAIL");

    printf("ref[0..7] : ");
    for (int i = 0; i < 8; i++) printf("%.3f ", ref[i]);
    printf("\nout[0..7] : ");
    for (int i = 0; i < 8; i++) printf("%.3f ", out[i]);
    printf("\n");

    delete[] Q; delete[] K; delete[] V; delete[] ref; delete[] out;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
}

int main()
{
    test_attention(BQ);   // seq_k=64  — régression sur le cas original
    test_attention(128);  // seq_k=128 — valide la boucle K tile
    return 0;
}