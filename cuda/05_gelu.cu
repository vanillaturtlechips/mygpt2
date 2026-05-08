/*
 * 05_gelu.cu — GELU 활성화 함수
 *
 * GELU(x) = x × 0.5 × (1 + tanh(√(2/π) × (x + 0.044715x³)))
 *
 * transformer FFN (Feed-Forward Network)에서 사용:
 *   FFN(x) = GELU(x × W1 + b1) × W2 + b2
 *
 * 원소별(element-wise) 연산 → 가장 단순한 병렬화
 * 스레드 하나가 원소 하나를 담당
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK 256
#define SQRT_2_PI 0.7978845608f   // sqrt(2/π)
#define COEFF     0.044715f


// =============================================================================
// 커널: GELU (원소별)
// =============================================================================

__global__ void gelu(float *X, float *Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = X[i];
    float inner = SQRT_2_PI * (x + COEFF * x * x * x);
    Y[i] = x * 0.5f * (1.0f + tanhf(inner));
}


// =============================================================================
// 근사 버전: x × σ(1.702x)  — 빠르지만 약간 덜 정확
// =============================================================================

__global__ void gelu_approx(float *X, float *Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = X[i];
    Y[i] = x / (1.0f + expf(-1.702f * x));
}


// =============================================================================
// CPU 기준
// =============================================================================

void gelu_cpu(float *X, float *Y, int n) {
    for (int i = 0; i < n; i++) {
        float x = X[i];
        float inner = SQRT_2_PI * (x + COEFF * x * x * x);
        Y[i] = x * 0.5f * (1.0f + tanhf(inner));
    }
}


// =============================================================================
// main
// =============================================================================

int main() {
    // GPT-2 FFN 중간 레이어: 512 토큰 × (768 × 4) = 512 × 3072
    int n = 512 * 3072;
    int bytes = n * sizeof(float);

    float *h_X   = (float*)malloc(bytes);
    float *h_Y   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    for (int i = 0; i < n; i++)
        h_X[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;   // [-3, 3]

    gelu_cpu(h_X, h_ref, n);

    float *d_X, *d_Y;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);

    int grid = (n + BLOCK - 1) / BLOCK;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ── 정확 버전 ─────────────────────────────────────────────────────────────
    cudaEventRecord(t0);
    gelu<<<grid, BLOCK>>>(d_X, d_Y, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_exact; cudaEventElapsedTime(&ms_exact, t0, t1);

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_Y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("[정확]  시간: %.3f ms  최대 오차: %.8f\n", ms_exact, max_err);

    // ── 근사 버전 ─────────────────────────────────────────────────────────────
    cudaEventRecord(t0);
    gelu_approx<<<grid, BLOCK>>>(d_X, d_Y, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_approx; cudaEventElapsedTime(&ms_approx, t0, t1);

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
    max_err = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_Y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("[근사]  시간: %.3f ms  최대 오차: %.8f\n", ms_approx, max_err);

    // GELU 동작 확인
    printf("\nGELU 샘플 값:\n");
    float samples[] = {-3, -1, 0, 1, 3};
    for (int k = 0; k < 5; k++) {
        float x = samples[k];
        float inner = SQRT_2_PI * (x + COEFF * x*x*x);
        float y = x * 0.5f * (1.0f + tanhf(inner));
        printf("  GELU(%.0f) = %.4f\n", x, y);
    }

    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y); free(h_ref);
    return 0;
}
