/*
 * 07_embedding.cu — Embedding lookup + Bias add + Residual add
 *
 * 세 연산 모두 간단한 메모리 접근 패턴이라 한 파일에 묶음.
 *
 * Embedding: 토큰 ID → 벡터
 *   out[i] = weight[id[i]]   (행 선택)
 *
 * Bias add: Linear 레이어 편향
 *   out[i][j] = x[i][j] + bias[j]
 *
 * Residual add: 잔차 연결
 *   out[i] = a[i] + b[i]   (벡터 덧셈과 동일)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK 256


// =============================================================================
// 커널 1: Embedding lookup
// weight: [vocab_size × d_model]
// ids:    [seq_len]  (토큰 ID)
// out:    [seq_len × d_model]
// =============================================================================

__global__ void embedding_lookup(float *weight, int *ids, float *out,
                                  int seq_len, int d_model) {
    int tok = blockIdx.x;               // 토큰 인덱스
    int dim = threadIdx.x;              // 차원 인덱스
    if (tok >= seq_len || dim >= d_model) return;

    int token_id = ids[tok];
    out[tok * d_model + dim] = weight[token_id * d_model + dim];
}


// =============================================================================
// 커널 2: Bias add
// x:    [rows × cols]
// bias: [cols]
// =============================================================================

__global__ void bias_add(float *x, float *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    x[idx] += bias[idx % cols];
}


// =============================================================================
// 커널 3: Residual add (element-wise)
// out[i] = a[i] + b[i]
// =============================================================================

__global__ void residual_add(float *a, float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}


// =============================================================================
// main
// =============================================================================

int main() {
    int vocab_size = 1000;
    int d_model    = 768;    // GPT-2
    int seq_len    = 16;

    // ── Embedding 테스트 ──────────────────────────────────────────────────────
    float *h_weight = (float*)malloc(vocab_size * d_model * sizeof(float));
    int   *h_ids    = (int*)  malloc(seq_len * sizeof(int));
    float *h_out    = (float*)malloc(seq_len * d_model * sizeof(float));

    for (int i = 0; i < vocab_size * d_model; i++)
        h_weight[i] = (float)i / (vocab_size * d_model);
    for (int i = 0; i < seq_len; i++)
        h_ids[i] = i * 10;   // 토큰 ID: 0, 10, 20, ...

    float *d_weight; int *d_ids; float *d_out;
    cudaMalloc(&d_weight, vocab_size * d_model * sizeof(float));
    cudaMalloc(&d_ids,    seq_len * sizeof(int));
    cudaMalloc(&d_out,    seq_len * d_model * sizeof(float));
    cudaMemcpy(d_weight, h_weight, vocab_size * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, h_ids, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // 블록 = 토큰 하나, 스레드 = 차원 하나 (d_model <= 1024 가정)
    embedding_lookup<<<seq_len, d_model>>>(d_weight, d_ids, d_out,
                                            seq_len, d_model);
    cudaMemcpy(h_out, d_out, seq_len * d_model * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 검증: out[0] == weight[ids[0]], out[1] == weight[ids[1]]
    int errors = 0;
    for (int t = 0; t < seq_len; t++) {
        int id = h_ids[t];
        for (int d = 0; d < d_model; d++) {
            float expected = h_weight[id * d_model + d];
            if (fabsf(h_out[t * d_model + d] - expected) > 1e-6f) errors++;
        }
    }
    printf("[Embedding]  seq=%d  d_model=%d  오류: %d\n",
           seq_len, d_model, errors);

    // ── Bias add 테스트 ───────────────────────────────────────────────────────
    int rows = 512, cols = 768;
    float *h_x    = (float*)malloc(rows * cols * sizeof(float));
    float *h_bias = (float*)malloc(cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) h_x[i] = 1.0f;
    for (int i = 0; i < cols; i++) h_bias[i] = (float)i;

    float *d_x, *d_bias;
    cudaMalloc(&d_x,    rows * cols * sizeof(float));
    cudaMalloc(&d_bias, cols * sizeof(float));
    cudaMemcpy(d_x,    h_x,    rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, cols * sizeof(float),        cudaMemcpyHostToDevice);

    int n = rows * cols;
    bias_add<<<(n + BLOCK-1)/BLOCK, BLOCK>>>(d_x, d_bias, rows, cols);
    cudaMemcpy(h_x, d_x, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    errors = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (fabsf(h_x[r*cols+c] - (1.0f + c)) > 1e-4f) errors++;
    printf("[Bias add]   rows=%d  cols=%d  오류: %d\n", rows, cols, errors);

    // ── Residual add 테스트 ───────────────────────────────────────────────────
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_c = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    residual_add<<<(n + BLOCK-1)/BLOCK, BLOCK>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_c[i] - 3.0f) > 1e-6f) errors++;
    printf("[Residual]   n=%d  오류: %d\n", n, errors);

    cudaFree(d_weight); cudaFree(d_ids); cudaFree(d_out);
    cudaFree(d_x); cudaFree(d_bias);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_weight); free(h_ids); free(h_out);
    free(h_x); free(h_bias);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
