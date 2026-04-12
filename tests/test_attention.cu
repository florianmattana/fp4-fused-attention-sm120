#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <common.h>

static constexpr int SEQ_Q = BQ;

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
                   float* Out,
                   int seq_q, int seq_k, int head_dim)
{
    float* Kt = new float[seq_k * head_dim];
    float* S  = new float[seq_q * seq_k];

    for (int i = 0; i < seq_k; i++)
        for (int j = 0; j < head_dim; j++)
            Kt[j * seq_k + i] = K[i * head_dim + j];

    cpu_matmul(Q, Kt, S, seq_q, seq_k, head_dim);

    float scale = 1.0f / sqrtf((float)head_dim);
    for (int i = 0; i < seq_q * seq_k; i++) S[i] *= scale;

    cpu_softmax_rows(S, seq_q, seq_k);
    cpu_matmul(S, V, Out, seq_q, head_dim, seq_k);

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

template<int HEAD_DIM>
void test_single_head(int seq_k)
{
    printf("\n=== single head  head_dim=%d  seq_q=%d  seq_k=%d ===\n",
           HEAD_DIM, SEQ_Q, seq_k);

    float* Q   = new float[SEQ_Q * HEAD_DIM];
    float* K   = new float[seq_k * HEAD_DIM];
    float* V   = new float[seq_k * HEAD_DIM];
    float* ref = new float[SEQ_Q * HEAD_DIM];
    float* out = new float[SEQ_Q * HEAD_DIM];

    srand(42);
    auto rnd = []{
        float vals[] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
        return vals[rand() % 5];
    };
    for (int i = 0; i < SEQ_Q * HEAD_DIM; i++) Q[i] = rnd();
    for (int i = 0; i < seq_k * HEAD_DIM; i++) K[i] = rnd();
    for (int i = 0; i < seq_k * HEAD_DIM; i++) V[i] = rnd();

    cpu_attention(Q, K, V, ref, SEQ_Q, seq_k, HEAD_DIM);

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ,   SEQ_Q * HEAD_DIM * sizeof(float));
    cudaMalloc(&dK,   seq_k * HEAD_DIM * sizeof(float));
    cudaMalloc(&dV,   seq_k * HEAD_DIM * sizeof(float));
    cudaMalloc(&dOut, SEQ_Q * HEAD_DIM * sizeof(float));
    cudaMemcpy(dQ, Q, SEQ_Q * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, seq_k * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, seq_k * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, SEQ_Q * HEAD_DIM * sizeof(float));

    fused_fp4_attention<HEAD_DIM><<<1, NUM_THREADS>>>(dQ, dK, dOut, dV,
                                                       SEQ_Q, seq_k, 1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("kernel error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaMemcpy(out, dOut, SEQ_Q * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    float cos_sim = cosine_similarity(ref, out, SEQ_Q * HEAD_DIM);
    printf("cosine : %.4f  %s\n", cos_sim, cos_sim >= 0.99f ? "PASS" : "FAIL");

    delete[] Q; delete[] K; delete[] V; delete[] ref; delete[] out;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
}

template<int HEAD_DIM>
void test_multi_head(int batch, int heads, int seq_k)
{
    printf("\n=== multi-head  head_dim=%d  batch=%d  heads=%d  seq_q=%d  seq_k=%d ===\n",
           HEAD_DIM, batch, heads, SEQ_Q, seq_k);

    int total_q = batch * heads * SEQ_Q * HEAD_DIM;
    int total_k = batch * heads * seq_k * HEAD_DIM;

    float* Q   = new float[total_q];
    float* K   = new float[total_k];
    float* V   = new float[total_k];
    float* ref = new float[total_q];
    float* out = new float[total_q];

    srand(42);
    auto rnd = []{
        float vals[] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
        return vals[rand() % 5];
    };
    for (int i = 0; i < total_q; i++) Q[i] = rnd();
    for (int i = 0; i < total_k; i++) K[i] = rnd();
    for (int i = 0; i < total_k; i++) V[i] = rnd();

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            int q_off = (b * heads + h) * SEQ_Q * HEAD_DIM;
            int k_off = (b * heads + h) * seq_k * HEAD_DIM;
            cpu_attention(Q + q_off, K + k_off, V + k_off,
                          ref + q_off, SEQ_Q, seq_k, HEAD_DIM);
        }
    }

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ,   total_q * sizeof(float));
    cudaMalloc(&dK,   total_k * sizeof(float));
    cudaMalloc(&dV,   total_k * sizeof(float));
    cudaMalloc(&dOut, total_q * sizeof(float));
    cudaMemcpy(dQ, Q, total_q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, total_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, total_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, total_q * sizeof(float));

    fused_fp4_attention<HEAD_DIM><<<batch * heads, NUM_THREADS>>>(
        dQ, dK, dOut, dV, SEQ_Q, seq_k, heads);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("kernel error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaMemcpy(out, dOut, total_q * sizeof(float), cudaMemcpyDeviceToHost);

    float cos_sim = cosine_similarity(ref, out, total_q);
    printf("cosine : %.4f  %s\n", cos_sim, cos_sim >= 0.99f ? "PASS" : "FAIL");

    delete[] Q; delete[] K; delete[] V; delete[] ref; delete[] out;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
}

int main()
{
    test_single_head<128>(BQ);
    test_single_head<128>(128);
    test_single_head<64>(BQ);
    test_single_head<64>(128);
    test_multi_head<128>(1, 4, 128);
    test_multi_head<128>(2, 4, 128);
    return 0;
}