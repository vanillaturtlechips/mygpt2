/*
 * 02_matmul.cu — 행렬곱 (타일 공유메모리 버전)
 *
 * C[M][N] = A[M][K] × B[K][N]
 *
 * 핵심 아이디어: 공유 메모리(Shared Memory)
 *   GPU 글로벌 메모리는 느림 (~400 사이클)
 *   블록 내 공유 메모리는 빠름 (~4 사이클)
 *   → 타일 단위로 공유 메모리에 올려놓고 계산
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE 16   // 타일 크기 (16×16 = 256 스레드/블록)


// =============================================================================
// 나이브 버전 — 스레드 하나가 C의 원소 하나를 계산
// =============================================================================

__global__ void matmul_naive(float *A, float *B, float *C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


// =============================================================================
// 타일 버전 — 공유 메모리로 글로벌 메모리 접근 횟수를 TILE배 줄임
// =============================================================================

__global__ void matmul_tiled(float *A, float *B, float *C,
                              int M, int K, int N) {
    // 블록 내 공유 메모리 — 한 타일씩 올려서 재사용
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // K 방향으로 TILE 크기만큼 슬라이딩
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        // 타일 로드: A에서 한 행 조각, B에서 한 열 조각
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE + threadIdx.x < K)
            ? A[row * K + t * TILE + threadIdx.x] : 0.0f;

        sB[threadIdx.y][threadIdx.x] =
            (t * TILE + threadIdx.y < K && col < N)
            ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;

        __syncthreads();   // 모든 스레드가 로드 완료될 때까지 대기

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   // 다음 타일 로드 전에 계산 완료 대기
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


// =============================================================================
// CPU 기준 행렬곱 (검증용)
// =============================================================================

void matmul_cpu(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}


// =============================================================================
// main
// =============================================================================

int main() {
    int M = 512, K = 512, N = 512;
    int bytesA = M * K * sizeof(float);
    int bytesB = K * N * sizeof(float);
    int bytesC = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    float *h_ref = (float*)malloc(bytesC);

    // 랜덤 초기화
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // CPU 기준값
    matmul_cpu(h_A, h_B, h_ref, M, K, N);

    // GPU 메모리
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);

    // ── 나이브 ───────────────────────────────────────────────────────────────
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_naive; cudaEventElapsedTime(&ms_naive, t0, t1);

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(h_C[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("[나이브] 시간: %.3f ms  최대 오차: %.6f\n", ms_naive, max_err);

    // ── 타일 ─────────────────────────────────────────────────────────────────
    cudaEventRecord(t0);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, t0, t1);

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(h_C[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("[타일]  시간: %.3f ms  최대 오차: %.6f\n", ms_tiled, max_err);
    printf("속도 향상: %.1fx\n", ms_naive / ms_tiled);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
