/*
 * 01_vector_add.cu — CUDA 첫 번째 커널
 *
 * 목표: C[i] = A[i] + B[i] 를 GPU에서 병렬로 계산
 *
 * CPU 버전:
 *   for (int i = 0; i < N; i++)
 *       C[i] = A[i] + B[i];     // 순서대로 하나씩
 *
 * GPU 버전:
 *   모든 i를 동시에 계산 — 스레드 하나가 원소 하나를 담당
 */

#include <stdio.h>
#include <stdlib.h>

#define N 1024      // 벡터 크기
#define BLOCK 256   // 블록당 스레드 수


// =============================================================================
// GPU 커널  __global__ = CPU에서 호출, GPU에서 실행
// =============================================================================

__global__ void vector_add(float *A, float *B, float *C, int n) {
    // 이 스레드가 담당하는 인덱스 계산
    // blockIdx.x  = 몇 번째 블록인지
    // blockDim.x  = 블록 하나에 스레드가 몇 개인지
    // threadIdx.x = 이 블록 안에서 몇 번째 스레드인지
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)          // 범위 초과 방지
        C[i] = A[i] + B[i];
}


// =============================================================================
// CPU (host) 코드
// =============================================================================

int main() {
    int bytes = N * sizeof(float);

    // ── CPU 메모리 할당 ──────────────────────────────────────────────────────
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)i * 2.0f;
    }

    // ── GPU 메모리 할당 ──────────────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // ── CPU → GPU 복사 ───────────────────────────────────────────────────────
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ── 커널 실행 ────────────────────────────────────────────────────────────
    // N=1024, BLOCK=256 → 블록 4개, 각 블록에 스레드 256개 → 총 1024 스레드
    int grid = (N + BLOCK - 1) / BLOCK;
    vector_add<<<grid, BLOCK>>>(d_A, d_B, d_C, N);

    // ── GPU → CPU 복사 ───────────────────────────────────────────────────────
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // ── 결과 확인 ────────────────────────────────────────────────────────────
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) errors++;
    }

    printf("벡터 크기  : %d\n", N);
    printf("블록 크기  : %d 스레드\n", BLOCK);
    printf("그리드 크기: %d 블록\n", grid);
    printf("총 스레드  : %d\n", grid * BLOCK);
    printf("오류       : %d\n", errors);
    printf("결과 샘플  : C[0]=%.0f  C[1]=%.0f  C[1023]=%.0f\n",
           h_C[0], h_C[1], h_C[1023]);

    // ── 메모리 해제 ──────────────────────────────────────────────────────────
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A);     free(h_B);     free(h_C);

    return 0;
}
