/*
 * libengine.cu — Transformer CUDA 커널 공유 라이브러리
 *
 * 컴파일:
 *   nvcc -shared -fPIC -o libengine.so libengine.cu -lm
 *
 * Python에서:
 *   import ctypes
 *   lib = ctypes.CDLL('./cuda/libengine.so')
 *
 * 모든 함수는 extern "C" — C 이름 맹글링으로 Python ctypes에서 호출 가능
 * GPU 포인터를 직접 받음 — 메모리 관리는 Python 측에서 처리
 */

#include <stdio.h>
#include <math.h>
#include <float.h>

#define TILE  16
#define BLOCK 256


// =============================================================================
// 내부 커널 (static — 외부 노출 안 함)
// =============================================================================

__global__ static void _matmul(float *A, float *B, float *C,
                                int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE-1)/TILE; t++) {
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t*TILE+threadIdx.x < K) ? A[row*K+t*TILE+threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t*TILE+threadIdx.y < K && col < N) ? B[(t*TILE+threadIdx.y)*N+col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

__global__ static void _softmax(float *X, float *Y, int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    float *x = X + row*cols, *y = Y + row*cols;
    float lmax = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) lmax = fmaxf(lmax, x[i]);
    smem[threadIdx.x] = lmax; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    float mx = smem[0];
    float lsum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) lsum += expf(x[i]-mx);
    smem[threadIdx.x] = lsum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float rs = smem[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[i] = expf(x[i]-mx)/rs;
}

__global__ static void _layernorm(float *X, float *Y, float *gamma, float *beta,
                                   int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    float *x = X+row*cols, *y = Y+row*cols;
    float ls = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ls += x[i];
    smem[threadIdx.x] = ls; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float mean = smem[0]/cols;
    float lv = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float d=x[i]-mean; lv+=d*d; }
    smem[threadIdx.x] = lv; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0]/cols + 1e-5f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = (x[i]-mean)*inv_std*gamma[i]+beta[i];
}

__global__ static void _gelu(float *X, float *Y, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    Y[i] = x*0.5f*(1.f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
}

__global__ static void _masked_softmax(float *S, int seq_len, float scale) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= seq_len) return;
    float *s = S+row*seq_len;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        s[i] = s[i]*scale;
        if (i > row) s[i] = -1e38f;
    }
    __syncthreads();
    float lmax = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) lmax = fmaxf(lmax, s[i]);
    smem[threadIdx.x] = lmax; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (threadIdx.x < st) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+st]);
        __syncthreads();
    }
    float mx = smem[0];
    float lsum = 0.f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) lsum += expf(s[i]-mx);
    smem[threadIdx.x] = lsum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (threadIdx.x < st) smem[threadIdx.x] += smem[threadIdx.x+st];
        __syncthreads();
    }
    float rs = smem[0];
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        s[i] = (s[i] <= -1e37f) ? 0.f : expf(s[i]-mx)/rs;
}

__global__ static void _embedding(float *W, int *ids, float *out,
                                   int seq_len, int d_model) {
    int tok = blockIdx.x, dim = threadIdx.x;
    if (tok >= seq_len || dim >= d_model) return;
    out[tok*d_model+dim] = W[ids[tok]*d_model+dim];
}

__global__ static void _add(float *a, float *b, float *out, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = a[i]+b[i];
}

__global__ static void _bias_add(float *x, float *bias, int rows, int cols) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < rows*cols) x[i] += bias[i%cols];
}


// =============================================================================
// 공개 C 인터페이스
// =============================================================================

extern "C" {

// ── 메모리 관리 ──────────────────────────────────────────────────────────────

void* engine_malloc(int bytes) {
    void *ptr; cudaMalloc(&ptr, bytes); return ptr;
}

void engine_free(void *ptr) { cudaFree(ptr); }

void engine_h2d(void *dst, void *src, int bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void engine_d2h(void *dst, void *src, int bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void engine_sync() { cudaDeviceSynchronize(); }

// ── 커널 래퍼 ────────────────────────────────────────────────────────────────

void engine_matmul(float *A, float *B, float *C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    _matmul<<<grid, block>>>(A, B, C, M, K, N);
}

void engine_softmax(float *X, float *Y, int rows, int cols) {
    _softmax<<<rows, BLOCK, BLOCK*sizeof(float)>>>(X, Y, rows, cols);
}

void engine_layernorm(float *X, float *Y, float *gamma, float *beta,
                      int rows, int cols) {
    _layernorm<<<rows, BLOCK, BLOCK*sizeof(float)>>>(X, Y, gamma, beta, rows, cols);
}

void engine_gelu(float *X, float *Y, int n) {
    _gelu<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(X, Y, n);
}

void engine_embedding(float *W, int *ids, float *out, int seq_len, int d_model) {
    _embedding<<<seq_len, d_model>>>(W, ids, out, seq_len, d_model);
}

void engine_add(float *a, float *b, float *out, int n) {
    _add<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(a, b, out, n);
}

void engine_bias_add(float *x, float *bias, int rows, int cols) {
    int n = rows*cols;
    _bias_add<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, bias, rows, cols);
}

// ── Attention (고수준) ───────────────────────────────────────────────────────
// Q, K, V, scores: 이미 GPU에 올라간 포인터
void engine_attention(float *Q, float *K, float *V,
                      float *out, float *scores,
                      int seq_len, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);
    dim3 block(TILE, TILE);

    dim3 gs((seq_len+TILE-1)/TILE, (seq_len+TILE-1)/TILE);
    _matmul<<<gs, block>>>(Q, K, scores, seq_len, d_k, seq_len);

    _masked_softmax<<<seq_len, BLOCK, BLOCK*sizeof(float)>>>(scores, seq_len, scale);

    dim3 go((d_k+TILE-1)/TILE, (seq_len+TILE-1)/TILE);
    _matmul<<<go, block>>>(scores, V, out, seq_len, seq_len, d_k);
}

} // extern "C"
