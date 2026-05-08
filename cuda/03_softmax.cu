/*
 * 03_softmax.cu — 행별 Softmax
 *
 * softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
 *
 * 핵심: max와 sum을 구하는 "병렬 리덕션(Parallel Reduction)"
 *   CPU: for 루프로 순서대로 합산 → O(N)
 *   GPU: 트리 구조로 절반씩 합산 → O(log N)
 *
 *   [a b c d e f g h]
 *    ↓
 *   [a+e b+f c+g d+h]
 *    ↓
 *   [a+c+e+g b+d+f+h]  (잘못됨, 개념만)
 *    ↓
 *   [합계]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define BLOCK 256   // 블록당 스레드 (행 하나당 블록 하나)


// =============================================================================
// 커널: 행별 Softmax
// 각 블록이 행 하나를 담당
// =============================================================================

__global__ void softmax(float *X, float *Y, int rows, int cols) {
    extern __shared__ float smem[];   // 동적 공유 메모리

    int row = blockIdx.x;             // 이 블록이 담당하는 행
    if (row >= rows) return;

    float *x = X + row * cols;
    float *y = Y + row * cols;

    // ── 1단계: 최댓값 구하기 (수치 안정성) ────────────────────────────────
    // 스레드마다 자기 담당 구간의 max를 먼저 구함
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, x[i]);

    smem[threadIdx.x] = local_max;
    __syncthreads();

    // 트리 리덕션으로 전체 max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x],
                                      smem[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = smem[0];

    // ── 2단계: exp 합계 구하기 ────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += expf(x[i] - row_max);

    smem[threadIdx.x] = local_sum;
    __syncthreads();

    // 트리 리덕션으로 전체 합
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = smem[0];

    // ── 3단계: 정규화 ─────────────────────────────────────────────────────
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - row_max) / row_sum;
}


// =============================================================================
// CPU 기준 softmax (검증)
// =============================================================================

void softmax_cpu(float *X, float *Y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = -1e38f;
        for (int c = 0; c < cols; c++) mx = fmaxf(mx, X[r*cols+c]);
        float s = 0;
        for (int c = 0; c < cols; c++) s += expf(X[r*cols+c] - mx);
        for (int c = 0; c < cols; c++) Y[r*cols+c] = expf(X[r*cols+c]-mx)/s;
    }
}


// =============================================================================
// main
// =============================================================================

int main() {
    // Attention 행렬 크기 시뮬레이션: seq_len=512, 512개 행
    int rows = 512, cols = 512;
    int bytes = rows * cols * sizeof(float);

    float *h_X   = (float*)malloc(bytes);
    float *h_Y   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    for (int i = 0; i < rows * cols; i++)
        h_X[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;   // [-2, 2]

    softmax_cpu(h_X, h_ref, rows, cols);

    float *d_X, *d_Y;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);

    // 행마다 블록 1개, 공유 메모리 = BLOCK개 float
    int smem = BLOCK * sizeof(float);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    softmax<<<rows, BLOCK, smem>>>(d_X, d_Y, rows, cols);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

    // 검증
    float max_err = 0;
    for (int i = 0; i < rows * cols; i++) {
        float e = fabsf(h_Y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    // 첫 번째 행 합계 = 1인지 확인
    float row_sum = 0;
    for (int c = 0; c < cols; c++) row_sum += h_Y[c];

    printf("행 크기    : %d × %d\n", rows, cols);
    printf("시간       : %.3f ms\n", ms);
    printf("최대 오차  : %.8f\n", max_err);
    printf("첫 행 합계 : %.6f (1.0이어야 함)\n", row_sum);

    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y); free(h_ref);
    return 0;
}
