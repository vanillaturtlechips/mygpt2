# ML/AI 면접 수학 공식 총정리 — Phase 1 · 2 · 3

> 신입 ML/AI 엔지니어 기준, 암기 우선순위 공식 모음

---

1 → 2 → 3 → 7 → 8 → 10
                ↓
           5 → 11 → 14
                ↓
                4 (병렬로 가볍게)


## Phase 1 — 수학 기초 (Math Foundations)

### 1. 선형대수 (Linear Algebra)

```
내적 (Dot product)
  a·b = Σ(aᵢ·bᵢ)

벡터 크기 (Magnitude / L2 norm)
  ||x|| = √(Σxᵢ²)

코사인 유사도 (Cosine similarity)
  cos(θ) = (a·b) / (||a||·||b||)

정사영 (Projection)
  proj_b(a) = (a·b / b·b) · b

그람-슈미트 직교화 (Gram-Schmidt)
  u1 = v1 / |v1|
  w2 = v2 - (v2·u1)u1
  u2 = w2 / |w2|

행렬 곱 크기 규칙
  (m×n) @ (n×p) = (m×p)

2×2 행렬식 (Determinant)
  det([[a,b],[c,d]]) = ad - bc

2×2 역행렬 (Inverse)
  A⁻¹ = (1/det) · [[d, -b], [-c, a]]

2D 회전 행렬 (Rotation matrix)
  R(θ) = [[cosθ, -sinθ],
           [sinθ,  cosθ]]

고유값/고유벡터 (Eigenvalue/Eigenvector)
  Av = λv
  특성방정식: λ² - (a+d)λ + (ad-bc) = 0   (2×2)

고유분해 (Eigendecomposition)
  A = V D V⁻¹   (V: 고유벡터 행렬, D: 대각 고유값 행렬)

특이값 분해 (SVD)
  A = U Σ Vᵀ
  저랭크 근사: Aₖ = Σᵢ₌₁ᵏ σᵢ·uᵢ·vᵢᵀ
  의사역행렬: A⁺ = V Σ⁺ Uᵀ

주성분 분석 (PCA)
  공분산: C = (1/(n-1)) · XᵀX
  설명 분산 비율: eigenvalue_k / Σ eigenvalues

정규방정식 (Normal equations)
  (XᵀX)w = Xᵀy
  Ridge: (XᵀX + λI)w = Xᵀy

조건수 (Condition number)
  κ = σ_max / σ_min
```

---

### 2. 노름 (Norms & Distances)

```
L1 (Manhattan):  ||x||₁ = Σ|xᵢ|
L2 (Euclidean):  ||x||₂ = √(Σxᵢ²)
L∞ (Chebyshev): ||x||∞ = max(|xᵢ|)

마할라노비스 거리 (Mahalanobis)
  d_M = √((x-y)ᵀ S⁻¹ (x-y))

자카드 유사도 (Jaccard similarity)
  J(A,B) = |A∩B| / |A∪B|
```

---

### 3. 미적분 (Calculus)

```
수치 미분 (Numerical derivative — central difference)
  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

주요 도함수
  d/dx(xⁿ)    = n·xⁿ⁻¹
  d/dx(x²)    = 2x
  d/dx(wx+b)  = w
  d/dx(eˣ)    = eˣ
  d/dx(ln x)  = 1/x
  d/dx(σ(x))  = σ(x)(1 - σ(x))

연쇄 법칙 (Chain rule)
  dy/dx = f'(g(x)) · g'(x)
  ∂L/∂x = (∂L/∂z) · (∂z/∂x)

테일러 전개 (Taylor expansion)
  f(x+h) = f(x) + f'(x)h + (1/2)f''(x)h² + ...

경사하강법 (Gradient descent update)
  w ← w - lr · ∇w

헤시안 & 뉴턴법 (Hessian & Newton's method)
  H[i][j] = ∂²f / (∂xᵢ∂xⱼ)
  w ← w - H⁻¹ · ∇f
```

---

### 4. 확률 (Probability)

```
조건부 확률 (Conditional probability)
  P(A|B) = P(A∩B) / P(B)

독립 (Independence)
  P(A∩B) = P(A)·P(B)

베이즈 정리 (Bayes' theorem)
  P(A|B) = P(B|A)·P(A) / P(B)

기댓값 & 분산 (Expectation & Variance)
  E[X] = Σ xᵢ·P(xᵢ)
  Var(X) = E[X²] - (E[X])²

정규분포 PDF (Normal distribution)
  f(x) = (1/√(2πσ²)) · exp(-(x-μ)² / (2σ²))

베르누이 (Bernoulli)
  Mean = p,  Var = p(1-p)

포아송 (Poisson)
  P(X=k) = (λᵏ·e⁻λ) / k!
  Mean = Var = λ

나이브 베이즈 (Naive Bayes)
  score(class) = P(class) · ΠP(featureᵢ|class)

라플라스 스무딩 (Laplace smoothing)
  P(word|class) = (count + 1) / (total + vocab_size)

베타 공액 사전분포 (Beta conjugate prior)
  Beta(a, b) + s 성공, f 실패 → Beta(a+s, b+f)
```

---

### 5. 정보 이론 (Information Theory)

```
자기 정보 (Self-information)
  I(x) = -log₂(p(x))

엔트로피 (Entropy)
  H(P) = -Σ p(x)·log(p(x))

교차 엔트로피 (Cross-entropy)
  H(P,Q) = -Σ p(x)·log(q(x))

KL 발산 (KL divergence)
  D_KL(P||Q) = Σ p(x)·log(p(x)/q(x))
             = H(P,Q) - H(P)

상호 정보량 (Mutual information)
  I(X;Y) = H(X) - H(X|Y)

퍼플렉서티 (Perplexity)
  PPL = exp(H(P,Q))

소프트맥스 (Softmax)
  softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
  수치 안정화: subtract max(x) before exp

Logsumexp 트릭
  log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
```

---

### 6. 통계 (Statistics)

```
기초 통계량
  mean = Σxᵢ / n
  sample var = Σ(xᵢ - x̄)² / (n-1)
  Pearson r = Σ(xᵢ-x̄)(yᵢ-ȳ) / (n·σx·σy)

t 검정 (t-test)
  t = (x̄ - μ₀) / (s / √n)

카이제곱 검정 (Chi-squared)
  χ² = Σ (obs - exp)² / exp

효과 크기 (Cohen's d)
  d = (μ₁ - μ₂) / pooled_std

다중 검정 보정 (Bonferroni)
  α_adj = α / m
```

---

### 7. 수치 안정성 (Numerical Stability)

```
그래디언트 클리핑 (Gradient clipping)
  if ||g|| > max_norm:
      g = g · (max_norm / ||g||)

안정적 소프트맥스
  max 값 뺀 후 exp 계산
```

---

### 8. 볼록 최적화 (Convex Optimization)

```
볼록 함수 조건 (Convexity test)
  f(tx + (1-t)y) ≤ t·f(x) + (1-t)·f(y)

라그랑지안 (Lagrangian)
  L(x, λ) = f(x) + λ·g(x)

KKT 조건
  ∇f + λ∇g = 0,  g(x) ≤ 0,  λ ≥ 0,  λ·g(x) = 0
```

---

### 9. 복소수 & 푸리에 (Complex Numbers & Fourier)

```
오일러 공식 (Euler's formula)
  e^(iθ) = cos(θ) + i·sin(θ)

이산 푸리에 변환 (DFT)
  X[k] = Σₙ x[n]·e^(-2πikn/N)

역 DFT (IDFT)
  x[n] = (1/N) Σₖ X[k]·e^(2πikn/N)

FFT 버터플라이 (twiddle factor)
  X[k] = E[k] + e^(-2πik/N)·O[k]

합성곱 정리 (Convolution theorem)
  x * h = IFFT(FFT(x)·FFT(h))

파르세발 정리 (Parseval's theorem)
  Σ|x[n]|² = (1/N)·Σ|X[k]|²

위치 인코딩 (Positional encoding — Transformer)
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

### 10. 그래프 이론 (Graph Theory)

```
그래프 라플라시안 (Graph Laplacian)
  L = D - A   (D: degree matrix, A: adjacency matrix)

GCN 레이어
  H^(l+1) = σ(D̂^(-1/2) · Â · D̂^(-1/2) · H^(l) · W^(l))
```

---

### 11. 확률 과정 (Stochastic Processes)

```
랜덤 워크 분산 (Random walk variance)
  Var(Sₙ) = n,  SD = √n

마르코프 정상 분포 (Markov stationary distribution)
  π · P = π

랑주뱅 동역학 (Langevin dynamics)
  x_{t+1} = x_t - dt·∇U + √(2T·dt)·z

확산 모델 순방향 (Diffusion forward process)
  x_t = √(αₜ)·x_{t-1} + √(1-αₜ)·ε
```

---

### 12. 샘플링 (Sampling)

```
역 CDF 샘플링 — 지수분포
  x = -ln(u) / λ,  u ~ Uniform(0,1)

재매개변수화 트릭 (Reparameterization trick — VAE)
  z = μ + σ·ε,  ε ~ N(0,1)

검벨-소프트맥스 (Gumbel-softmax)
  yᵢ = exp((log(pᵢ) + gᵢ) / τ) / Σ exp((log(pⱼ) + gⱼ) / τ)
```

---

## Phase 2 — ML 기초 (Machine Learning Fundamentals)

### 1. 선형 회귀 (Linear Regression)

```
모델
  ŷ = wx + b  (단변량)
  ŷ = Xw + b  (다변량)

MSE 손실
  MSE = (1/n)·Σ(y - ŷ)²

그래디언트
  dL/dw = (2/n)·Σ(ŷ - y)·x
  dL/db = (2/n)·Σ(ŷ - y)

정규방정식
  w = (XᵀX)⁻¹·Xᵀy

Ridge (L2 규제)
  L = MSE + λ·Σwᵢ²
  w = (XᵀX + λI)⁻¹·Xᵀy

Lasso (L1 규제)
  L = MSE + λ·Σ|wᵢ|
  → soft-thresholding (좌표하강법)
```

---

### 2. 로지스틱 회귀 (Logistic Regression)

```
시그모이드
  σ(z) = 1 / (1 + e^(-z))

Binary cross-entropy
  BCE = -(y·log(p) + (1-y)·log(1-p))

그래디언트
  dL/dw = (p - y)·x

다중 클래스 → Softmax + CCE
  CCE = -Σ yᵢ·log(pᵢ)
```

---

### 3. 결정 트리 (Decision Tree)

```
정보 이득 (Information Gain)
  IG = H(parent) - Σ (|child| / |parent|)·H(child)

지니 불순도 (Gini impurity)
  Gini = 1 - Σ pᵢ²
```

---

### 4. SVM (Support Vector Machine)

```
결정 경계 마진
  margin = 2 / ||w||

힌지 손실 (Hinge loss)
  L = max(0, 1 - y·(w·x + b))

쌍대 목적함수 (Dual objective)
  max Σαᵢ - 0.5·ΣΣ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
  s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0

RBF 커널
  K(x, y) = exp(-γ·||x - y||²)

Polynomial 커널
  K(x, y) = (x·y + c)^d
```

---

### 5. K-최근접 이웃 (KNN)

```
유클리드 거리
  d(x, y) = √(Σ(xᵢ - yᵢ)²)

분류: 다수결 (majority vote)
회귀:  평균 (mean of k neighbors)
```

---

### 6. k-평균 군집화 (k-Means)

```
목적함수
  argmin Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²

갱신 규칙
  μᵢ = (1/|Cᵢ|)·Σ_{x∈Cᵢ} x
```

---

### 7. 특성 엔지니어링 (Feature Engineering)

```
표준화 (Standardization)
  z = (x - μ) / σ

Min-max 정규화
  x_norm = (x - x_min) / (x_max - x_min)

Z-점수 이상 탐지
  Z = (x - μ) / σ   → |Z| > 3 이면 이상치
```

---

### 8. 모델 평가 (Model Evaluation)

```
혼동 행렬 기반 지표
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2·(Precision·Recall) / (Precision + Recall)

AUC-ROC
  TPR = TP / (TP + FN),  FPR = FP / (FP + TN)

교차 검증 (k-Fold CV)
  score = mean(score over k folds)
```

---

### 9. 편향-분산 트레이드오프 (Bias-Variance Tradeoff)

```
MSE 분해
  MSE = Bias² + Variance + Irreducible Noise

언더피팅 → 높은 bias
오버피팅  → 높은 variance
```

---

### 10. 앙상블 (Ensemble)

```
배깅 (Bagging — 예: Random Forest)
  ŷ = (1/T)·Σᵢ ŷᵢ

부스팅 (Boosting — 예: GBM, AdaBoost)
  F_m(x) = F_{m-1}(x) + η·h_m(x)

AdaBoost 가중치 갱신
  wᵢ ← wᵢ·exp(-αᵢ·yᵢ·h(xᵢ))
  αᵢ = 0.5·ln((1-errᵢ)/errᵢ)
```

---

### 11. 시계열 (Time Series)

```
AR(p) 모델
  yₜ = c + Σᵢ₌₁ᵖ φᵢ·yₜ₋ᵢ + εₜ

ARIMA: AR + 차분 + MA 결합
```

---

## Phase 3 — 딥러닝 핵심 (Deep Learning Core)

### 1. 퍼셉트론 & 순전파 (Perceptron & Forward Pass)

```
선형 결합
  z = Σ wᵢ·xᵢ + b  =  W·x + b

퍼셉트론 계단 함수
  output = 1 if z ≥ 0 else 0

퍼셉트론 학습 규칙
  wᵢ ← wᵢ + lr·error·xᵢ
  b   ← b  + lr·error

다층 순전파 (2층 예시)
  z1 = W1·x + b1,  a1 = σ(z1)
  z2 = W2·a1 + b2, a2 = σ(z2)
```

---

### 2. 역전파 (Backpropagation)

```
연쇄 법칙
  dL/dW1 = dL/dz1 · x
  dL/dz2 = 2(a2 - y)·a2(1 - a2)    (MSE + sigmoid 출력)

일반 규칙
  dL/dW^(l) = δ^(l) · (a^(l-1))ᵀ
  δ^(l)     = (W^(l+1))ᵀ δ^(l+1) ⊙ σ'(z^(l))
```

---

### 3. 활성화 함수 & 도함수 (Activation Functions)

```
Sigmoid
  σ(x) = 1 / (1 + e^(-x))
  σ'(x) = σ(x)·(1 - σ(x))         ← 최대 0.25

Tanh
  tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
  tanh'(x) = 1 - tanh²(x)

ReLU
  f(x) = max(0, x)
  f'(x) = 1 if x > 0 else 0

Leaky ReLU
  f(x) = x if x > 0 else α·x    (α ≈ 0.01)

GELU
  f(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

Swish
  f(x) = x·σ(x)

Softmax
  softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

---

### 4. 손실 함수 (Loss Functions)

```
MSE
  L = (1/n)·Σ(ŷ - y)²
  ∂L/∂ŷ = (2/n)·(ŷ - y)

Binary Cross-Entropy (BCE)
  L = -(y·log(p) + (1-y)·log(1-p))
  ∂L/∂p = -(y/p) + (1-y)/(1-p)

Categorical Cross-Entropy (CCE)
  L = -Σ yᵢ·log(pᵢ)

레이블 스무딩 (Label smoothing)
  y_smooth = (1-α)·one_hot + α/K

InfoNCE (대조 학습)
  L = -log[ exp(sim(zᵢ,zⱼ)/τ) / Σₖ exp(sim(zᵢ,zₖ)/τ) ]

Triplet loss
  L = max(0, d(a,p) - d(a,n) + margin)

Focal loss (클래스 불균형)
  L = -α·(1-pₜ)^γ·log(pₜ)

Huber loss (이상치 강건)
  L = { 0.5·δ²           if |δ| ≤ 1
        |δ| - 0.5        if |δ| > 1 }
```

---

### 5. 옵티마이저 (Optimizers)

```
SGD
  w ← w - lr·∇w

Momentum
  m ← β·m + ∇w
  w ← w - lr·m

RMSProp
  s ← β·s + (1-β)·(∇w)²
  w ← w - lr·∇w / (√s + ε)

Adam (β₁=0.9, β₂=0.999, ε=1e-8)
  m  ← β₁·m + (1-β₁)·∇w
  v  ← β₂·v + (1-β₂)·(∇w)²
  m̂  = m / (1-β₁ᵗ)              ← bias correction
  v̂  = v / (1-β₂ᵗ)
  w  ← w - lr·m̂ / (√v̂ + ε)

AdamW (weight decay 분리)
  w ← w - lr·m̂ / (√v̂ + ε) - lr·λ·w

그래디언트 클리핑
  if ||g|| > max_norm:
      g = g · (max_norm / ||g||)
```

---

### 6. 정규화 (Regularization)

```
Dropout (역 드롭아웃 — inverted dropout)
  output = activation(z)·mask / (1-p)
  추론 시: mask 제거, 스케일 보정 불필요

L2 규제
  L_total = L_task + (λ/2)·Σ wᵢ²
  → gradient에 λ·w 추가
```

---

### 7. 정규화 레이어 (Normalization Layers)

```
Batch Normalization
  μ_B  = (1/B)·Σ xᵢ
  σ²_B = (1/B)·Σ (xᵢ - μ_B)²
  x̂   = (x - μ_B) / √(σ²_B + ε)
  y    = γ·x̂ + β              ← 학습 가능 파라미터

Layer Normalization
  (동일 공식, 배치 축 대신 feature 축으로 계산)

RMSNorm
  rms = √((1/D)·Σ xⱼ²)
  y   = γ · x / rms
```

---

### 8. 가중치 초기화 (Weight Initialization)

```
Xavier 초기화 (tanh/sigmoid)
  Var(w) = 2 / (fan_in + fan_out)
  w ~ N(0, √(2/(fan_in + fan_out)))

Kaiming/He 초기화 (ReLU)
  Var(w) = 2 / fan_in
  w ~ N(0, √(2/fan_in))

분산 전파 원리
  Var(z) = fan_in · Var(w) · Var(x)

GPT-2 잔차 연결 스케일링
  가중치 × (1 / √(2N))   (N = 레이어 수)
```

---

### 9. 학습률 스케줄 (Learning Rate Schedules)

```
코사인 어닐링 (Cosine annealing)
  lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))

Warmup + 코사인 감쇠
  0 ~ warmup_steps: 선형 상승
  warmup_steps ~ T:  코사인 감쇠

Step decay
  lr ← lr · γ  (매 k 에폭마다)
```

---

### 10. 기울기 소실/폭발 (Vanishing/Exploding Gradient)

```
Sigmoid 기울기 소실
  0.25^N ≈ 10⁻⁶  (10 레이어 기준)

완화 방법
  - ReLU/GELU 사용 (기울기 포화 없음)
  - Batch/Layer Norm
  - Residual connections (identity shortcut)
  - Gradient clipping (폭발 방지)
```

---

## 빠른 암기 카드 (Quick Reference)

| 이름 | 공식 |
|------|------|
| Softmax | `exp(xᵢ) / Σexp(xⱼ)` |
| BCE | `-(y·log p + (1-y)·log(1-p))` |
| Adam update | `w -= lr·m̂/(√v̂+ε)` |
| Dropout | `output = act(z)·mask/(1-p)` |
| BatchNorm | `x̂=(x-μ)/√(σ²+ε)`, `y=γx̂+β` |
| Xavier | `w~N(0,√(2/(fi+fo)))` |
| Kaiming | `w~N(0,√(2/fan_in))` |
| Cosine LR | `lr_min+0.5(lr_max-lr_min)(1+cos(πt/T))` |
| Chain rule | `dL/dx = (dL/dz)·(dz/dx)` |
| KL divergence | `Σp·log(p/q)` |
| Cross-entropy | `-Σp·log(q)` |
| SVD | `A=UΣVᵀ` |
| Normal eq. | `(XᵀX)w=Xᵀy` |
| F1 score | `2PR/(P+R)` |
| Bias-var | `MSE=Bias²+Var+Noise` |
| RBF kernel | `exp(-γ‖x-y‖²)` |
| Hinge loss | `max(0, 1-y·f(x))` |
| RMSNorm | `y=γ·x/√(mean(x²))` |
| InfoNCE | `-log[exp(sim/τ)/Σexp(sim/τ)]` |
| Focal loss | `-α(1-pₜ)^γ·log(pₜ)` |

---

*생성일: 2026-05-13 | 대상: 신입 ML/AI 엔지니어 면접 준비*
