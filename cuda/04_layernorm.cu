/*
 * 04_layernorm.cu — Layer Normalization
 *
 * y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * transformer.py의 모든 레이어 전후에 들어가는 연산.
 * Softmax와 마찬가지로 행별 리덕션이 필요함.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK 256
#define EPS 1e-5f


// =============================================================================
// 커널: 행별 LayerNorm
// =============================================================================

__global__ void layernorm(float *X, float *Y,
                           float *gamma, float *beta,
                           int rows, int cols) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if (row >= rows) return;

    float *x = X + row * cols;
    float *y = Y + row * cols;

    // ── 1단계: 평균 ──────────────────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += x[i];

    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = smem[0] / cols;

    // ── 2단계: 분산 ──────────────────────────────────────────────────────────
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    smem[threadIdx.x] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float var = smem[0] / cols;
    float inv_std = rsqrtf(var + EPS);   // 1 / sqrt(var + eps)

    // ── 3단계: 정규화 + 스케일/시프트 ───────────────────────────────────────
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}


// =============================================================================
// CPU 기준 (검증)
// =============================================================================

void layernorm_cpu(float *X, float *Y, float *gamma, float *beta,
                   int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mean = 0, var = 0;
        for (int c = 0; c < cols; c++) mean += X[r*cols+c];
        mean /= cols;
        for (int c = 0; c < cols; c++) {
            float d = X[r*cols+c] - mean;
            var += d * d;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + EPS);
        for (int c = 0; c < cols; c++)
            Y[r*cols+c] = (X[r*cols+c] - mean) * inv_std * gamma[c] + beta[c];
    }
}


// =============================================================================
// main
// =============================================================================

int main() {
    // GPT-2 hidden dim = 768, seq_len = 512
    int rows = 512, cols = 768;
    int bytes    = rows * cols * sizeof(float);
    int bytes_gb = cols * sizeof(float);

    float *h_X     = (float*)malloc(bytes);
    float *h_Y     = (float*)malloc(bytes);
    float *h_ref   = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(bytes_gb);
    float *h_beta  = (float*)malloc(bytes_gb);

    for (int i = 0; i < rows * cols; i++)
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < cols; i++) {
        h_gamma[i] = 1.0f;   // 초기값: 스케일 1
        h_beta[i]  = 0.0f;   // 초기값: 시프트 0
    }

    layernorm_cpu(h_X, h_ref, h_gamma, h_beta, rows, cols);

    float *d_X, *d_Y, *d_gamma, *d_beta;
    cudaMalloc(&d_X,     bytes);
    cudaMalloc(&d_Y,     bytes);
    cudaMalloc(&d_gamma, bytes_gb);
    cudaMalloc(&d_beta,  bytes_gb);
    cudaMemcpy(d_X,     h_X,     bytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, bytes_gb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  bytes_gb, cudaMemcpyHostToDevice);

    int smem = BLOCK * sizeof(float);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    layernorm<<<rows, BLOCK, smem>>>(d_X, d_Y, d_gamma, d_beta, rows, cols);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < rows * cols; i++) {
        float e = fabsf(h_Y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    // 정규화 확인: 첫 행 평균≈0, 분산≈1
    float m = 0, v = 0;
    for (int c = 0; c < cols; c++) m += h_Y[c];
    m /= cols;
    for (int c = 0; c < cols; c++) v += (h_Y[c]-m)*(h_Y[c]-m);
    v /= cols;

    printf("크기       : %d × %d\n", rows, cols);
    printf("시간       : %.3f ms\n", ms);
    printf("최대 오차  : %.8f\n", max_err);
    printf("첫 행 평균 : %.6f (≈0)\n", m);
    printf("첫 행 분산 : %.6f (≈1)\n", v);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_X); free(h_Y); free(h_ref); free(h_gamma); free(h_beta);
    return 0;
}
