/*
 * 06_attention.cu — Scaled Dot-Product Attention
 *
 * Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
 *
 * 지금까지 만든 커널들의 조합:
 *   matmul_tiled  → Q × Kᵀ
 *   softmax       → 각 행을 정규화
 *   matmul_tiled  → scores × V
 *
 * GPT-2 기준:
 *   d_model = 768, n_heads = 12
 *   d_k = 768 / 12 = 64  (헤드당 차원)
 *   seq_len = 512
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define TILE  16
#define BLOCK 256


// =============================================================================
// 커널 1: 행렬곱 (02_matmul.cu 타일 버전 재사용)
// =============================================================================

__global__ void matmul_tiled(float *A, float *B, float *C,
                              int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE-1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t*TILE + threadIdx.x < K)
            ? A[row * K + t*TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (t*TILE + threadIdx.y < K && col < N)
            ? B[(t*TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}


// =============================================================================
// 커널 2: 스케일 + 마스킹 + Softmax (인과 마스크 포함)
// =============================================================================

__global__ void masked_softmax(float *scores, int seq_len, float scale) {
    extern __shared__ float smem[];

    int row = blockIdx.x;   // 쿼리 토큰 인덱스
    if (row >= seq_len) return;

    float *s = scores + row * seq_len;

    // ── 스케일 + 인과 마스크 적용 ────────────────────────────────────────────
    // 미래 토큰에 -inf를 넣어 attention 차단
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        s[i] = s[i] * scale;
        if (i > row) s[i] = -1e38f;   // 인과 마스크
    }
    __syncthreads();

    // ── max 리덕션 ────────────────────────────────────────────────────────────
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_max = fmaxf(local_max, s[i]);
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x],
                                      smem[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = smem[0];

    // ── sum 리덕션 ────────────────────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_sum += expf(s[i] - row_max);
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = smem[0];

    // ── 정규화 ────────────────────────────────────────────────────────────────
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        s[i] = (s[i] <= -1e37f) ? 0.0f : expf(s[i] - row_max) / row_sum;
}


// =============================================================================
// Attention 전체 흐름
// =============================================================================

void attention(float *d_Q, float *d_K, float *d_V, float *d_out,
               float *d_scores,
               int seq_len, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);

    dim3 block(TILE, TILE);

    // ── Q × Kᵀ → scores [seq_len × seq_len] ─────────────────────────────────
    // Kᵀ: K를 전치하지 않고 인덱스 접근 순서를 바꿔서 처리
    dim3 grid_s((seq_len + TILE-1)/TILE, (seq_len + TILE-1)/TILE);
    matmul_tiled<<<grid_s, block>>>(d_Q, d_K, d_scores,
                                    seq_len, d_k, seq_len);

    // ── 스케일 + 마스킹 + Softmax ────────────────────────────────────────────
    int smem = BLOCK * sizeof(float);
    masked_softmax<<<seq_len, BLOCK, smem>>>(d_scores, seq_len, scale);

    // ── scores × V → output [seq_len × d_k] ─────────────────────────────────
    dim3 grid_o((d_k + TILE-1)/TILE, (seq_len + TILE-1)/TILE);
    matmul_tiled<<<grid_o, block>>>(d_scores, d_V, d_out,
                                    seq_len, seq_len, d_k);
}


// =============================================================================
// CPU 기준 Attention (검증)
// =============================================================================

void attention_cpu(float *Q, float *K, float *V, float *out,
                   int seq_len, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);
    float *scores = (float*)malloc(seq_len * seq_len * sizeof(float));

    // Q × Kᵀ
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < seq_len; j++) {
            float s = 0;
            for (int k = 0; k < d_k; k++)
                s += Q[i*d_k+k] * K[j*d_k+k];
            scores[i*seq_len+j] = (j > i) ? -1e38f : s * scale;
        }

    // Softmax
    for (int i = 0; i < seq_len; i++) {
        float mx = -1e38f;
        for (int j = 0; j < seq_len; j++) mx = fmaxf(mx, scores[i*seq_len+j]);
        float s = 0;
        for (int j = 0; j < seq_len; j++) s += expf(scores[i*seq_len+j]-mx);
        for (int j = 0; j < seq_len; j++) {
            float v = scores[i*seq_len+j];
            scores[i*seq_len+j] = (v <= -1e37f) ? 0.0f : expf(v-mx)/s;
        }
    }

    // scores × V
    for (int i = 0; i < seq_len; i++)
        for (int k = 0; k < d_k; k++) {
            float s = 0;
            for (int j = 0; j < seq_len; j++)
                s += scores[i*seq_len+j] * V[j*d_k+k];
            out[i*d_k+k] = s;
        }
    free(scores);
}


// =============================================================================
// main
// =============================================================================

int main() {
    int seq_len = 128;    // 시연용 (512면 GTX 1060 VRAM 부족 가능)
    int d_k     = 64;     // GPT-2: 768 / 12 heads

    int bytes_qkv    = seq_len * d_k * sizeof(float);
    int bytes_scores = seq_len * seq_len * sizeof(float);

    float *h_Q   = (float*)malloc(bytes_qkv);
    float *h_K   = (float*)malloc(bytes_qkv);
    float *h_V   = (float*)malloc(bytes_qkv);
    float *h_out = (float*)malloc(bytes_qkv);
    float *h_ref = (float*)malloc(bytes_qkv);

    for (int i = 0; i < seq_len * d_k; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX) * 0.1f;
        h_K[i] = ((float)rand()/RAND_MAX) * 0.1f;
        h_V[i] = ((float)rand()/RAND_MAX) * 0.1f;
    }

    attention_cpu(h_Q, h_K, h_V, h_ref, seq_len, d_k);

    float *d_Q, *d_K, *d_V, *d_out, *d_scores;
    cudaMalloc(&d_Q,      bytes_qkv);
    cudaMalloc(&d_K,      bytes_qkv);
    cudaMalloc(&d_V,      bytes_qkv);
    cudaMalloc(&d_out,    bytes_qkv);
    cudaMalloc(&d_scores, bytes_scores);
    cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    attention(d_Q, d_K, d_V, d_out, d_scores, seq_len, d_k);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_out, d_out, bytes_qkv, cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < seq_len * d_k; i++) {
        float e = fabsf(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("seq_len    : %d\n", seq_len);
    printf("d_k        : %d\n", d_k);
    printf("시간       : %.3f ms\n", ms);
    printf("최대 오차  : %.8f\n", max_err);
    printf("인과 마스크: 미래 토큰 차단 적용됨\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_out); cudaFree(d_scores);
    free(h_Q); free(h_K); free(h_V); free(h_out); free(h_ref);
    return 0;
}
