# 학습 로그

매일 종료 전에 짧게 기록합니다. 동료 코드나 AI의 구현 제안을 보기 전에는 내가 시도한 내용을 먼저 남기세요.

## Day N - YYYY-MM-DD

### 오늘 직접 구현한 것

-

### 오늘 확인한 shape와 수식

- 입력/출력 shape:
- 손으로 확인한 계산:
- 반드시 성립해야 하는 조건:

### 막힌 지점과 해결 과정

- 증상:
- 내가 세운 가설:
- 시도 1:
- 시도 2:
- 해결 또는 다음 확인 항목:

### 참고 기록

| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
|  |  |  |

### 코드 없이 답할 질문 3개

1.
2.
3.

---

## Day 1 - 2026-11-27

### 오늘 직접 구현한 것

- `Affine.forward`: 입력 `x`를 `self.x`에 저장하고 `x @ W + b`를 반환
- `ReLU.forward`: `x <= 0`인 위치를 `self.mask`(bool 배열)에 저장하고, 복사본에서 그 위치를 0으로
- `SoftmaxWithLoss.forward`: 행별 max-shift → exp → 행 합으로 나눠 확률 → 정답 클래스 확률 clip → `-log` → batch 평균

### 오늘 확인한 shape와 수식

- 입력/출력 shape:
  - Affine: `x (N, 784) @ W (784, H) + b (H,) → (N, H)`. `b`는 `(1, H)`로 broadcast되어 모든 행에 같은 값이 더해진다
  - ReLU: shape가 바뀌지 않는다. `mask`도 `x`와 같은 shape
  - Softmax: `logits (N, C) → y_prob (N, C)`, 정답 확률 `(N,)`, loss는 스칼라
  - selfcheck trace: `(2, 784) → Affine (2, 16) → ReLU (2, 16) → Affine (2, 10)`
- 손으로 확인한 계산:
  - logits가 전부 0 → 확률이 전부 1/10 → loss = −ln(0.1) = ln 10 = 2.3026
  - `[1000, 1001, 1002]` → shift `[-2, -1, 0]` → 확률 `[0.090, 0.245, 0.665]` → 정답이 2일 때 loss = −ln 0.665 = 0.4076
  - 정답 확률 `[0.7, 0.8, 0.3]`: 샘플마다 −log 후 평균 = 0.595, 평균 후 −log = 0.511 → 순서가 바뀌면 값이 다르다
- 반드시 성립해야 하는 조건:
  - `y_prob` 각 행의 합 = 1, 모든 값이 0 이상 1 이하
  - ReLU forward 뒤에도 입력 `x`는 바뀌지 않는다
  - 예측이 좋아지면(정답 logit이 커지면) loss는 작아진다

### 막힌 지점과 해결 과정

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| pytest에서 `ModuleNotFoundError: numpy` | conda 환경이 아니라 시스템 Python 3.14로 실행됐다 | `conda activate mnist-nn` 후 `python -m pytest` |
| `(4,10)`과 `(4,)` broadcast 에러, `(3,3)`에서는 행 합이 1이 아님 | `axis=1`만 쓰면 `(N,)`이 되고, 뒤쪽 축(C)에 맞춰져서 열 방향으로 빠진다 | `keepdims=True`로 `(N, 1)` 유지 |
| loss가 0.1(정답 확률 평균)로 나옴 | clip과 평균 사이의 `-log`를 빠뜨렸다 | clip → `-log` → 평균 순서로 |

### 참고 기록

| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| AI (Claude Code) | Affine 식의 원소 단위 유도, `y_prob[np.arange(N), y_true]` 인덱싱, `-log`를 쓰는 이유 | 작은 배열로 직접 출력 비교, `pytest` Day 1 테스트 13개 통과, `selfcheck --day 1 --trace` |
| 과제 docstring | Softmax 구현 순서 1~4, clip을 쓰는 이유 | `test_softmax_with_loss.py` forward 테스트 6개 통과 |

---

## Day 2 - 2026-11-27

### 오늘 직접 구현한 것

- `SoftmaxWithLoss.backward`: `y_prob` 복사본의 정답 칸(`[np.arange(N), y_true]`)에서 1을 빼고 N으로 나눔
- `ReLU.backward`: forward에서 저장한 `mask` 위치의 `dout`을 0으로
- `Affine.backward`: `dW = x.T @ dout`, `db = dout.sum(axis=0)`, `dx = dout @ W.T`

### 오늘 확인한 shape와 수식

- 큰 그림: 각 층은 `dout`(= ∂L/∂출력)을 받아 `dx`(= ∂L/∂입력)를 돌려준다. ∂L/∂입력 = ∂L/∂출력 × ∂출력/∂입력이라 각 층은 자기 국소 미분만 알면 된다
- 입력/출력 shape:
  - Softmax: `dx (N, C)`. L = −z_y + log Σ e^{z_j}를 z_k로 미분하면 p_k − t_k이고, forward에서 batch 평균을 냈으니 1/N이 곱해진다
  - ReLU: 국소 미분은 x > 0이면 1, x ≤ 0이면 0. `dx`는 `dout`에서 mask 자리만 0, shape 그대로
  - Affine: `dW (D, H) = x.T (D, N) @ dout (N, H)`, `db (H,) = dout`을 샘플 방향으로 합, `dx (N, D) = dout (N, H) @ W.T (H, D)`
  - 스칼라로 보면 y = x·w + b에서 ∂L/∂w = dout·x, ∂L/∂b = dout·1, ∂L/∂x = dout·w. 행렬에서는 shape가 맞도록 `.T`가 붙는다
  - b는 broadcast로 N개 샘플에 똑같이 더해졌으니, 기울기도 N개를 다 더한다 (평균 아님. 1/N은 Softmax backward에서 이미 나눴다)
- 손으로 확인한 계산:
  - Softmax: logits `zeros((2, 3))`, 정답 `[0, 1]` → `[[1/3 − 1, 1/3, 1/3], [1/3, 1/3 − 1, 1/3]] / 2`
  - Affine (`test_affine.py` 예시): dW[0, 0] = 1·1 + 3·2 = 7, dx[0, 0] = 1·1 + 0·2 + (−1)·3 = −2, db = dout 세로 합 = `[3, 1, −1]`
  - ReLU: x = `[[-1, 2], [0, -3]]`, dout = `[[5, 7], [9, 11]]` → dx = `[[0, 7], [0, 0]]`
- 반드시 성립해야 하는 조건:
  - `dW`, `db`, `dx`의 shape는 각각 `W`, `b`, `x`와 같다
  - Softmax `dx`의 각 행 합은 0 (확률 합 1 − one-hot 합 1)
  - gradient check: 수치 미분 (L(w + ε) − L(w − ε)) / 2ε 와 backward 값이 같아야 한다. 허용 1e-6, 직접 비교했더니 W1·b1·W2·b2 모두 상대 오차 1e-11 수준
  - 테스트 출력의 "상대 오차 0.00e+00"은 차이가 `atol`(1e-8) 이하라 0으로 기록된 것이지, 완전히 같다는 뜻은 아니다
  - ReLU backward는 `dout`을 제자리에서 바꾼다. 지금 네트워크에서는 `dout`이 매번 새로 만들어진 배열이라 문제없다

### 막힌 지점과 해결 과정

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| Softmax backward에서 `(4,10)`과 `(4,)` broadcast 에러, N == C면 조용히 틀림 | `y_true`는 라벨 번호 `(N,)`이지 one-hot `(N, C)`가 아니다 | one-hot을 빼는 건 정답 칸에서만 1을 빼는 것과 같다 |
| `y_prob[self.y_true] - 1`에서 IndexError | 인덱스를 하나만 주면 행 번호로 읽는다. `- 1`은 모든 칸에서 뺀다 | `[np.arange(N), y_true]`로 행과 열을 둘 다 지정 |

### 참고 기록

| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| AI (Claude Code) | Softmax + Cross Entropy 미분 유도, 스칼라 chain rule, 연쇄 대입, `.T` 사용법, gradient check의 `atol` 의미 | `test_softmax_with_loss.py` 11개, `test_relu.py` 5개, `test_affine.py` 8개 통과, `selfcheck --day 2` 26 passed, 바닥값 없이 직접 비교한 상대 오차 1e-11 |
| 과제 docstring·테스트 | Softmax backward 순서 1~3, `test_affine.py`의 손계산 숫자, gradient check 실패 메시지의 범위 힌트 | 같은 숫자로 손계산한 값과 테스트 기대값 비교 |

---

## Day 3 - 2026-11-27

### 오늘 직접 구현한 것

- `SGD.update`: `params`를 for문으로 돌면서 key마다 `lr * grad`를 `-=`로 빼서 제자리 갱신
- `train()`: 미니배치마다 forward로 logits → criterion으로 loss → `criterion.backward()`의 반환값을 `model.backward()`에 dout으로 넘김 → `model.params`를 `model.grads`로 update
- MNIST baseline 1회 (hidden 128, SGD lr 0.1, batch 128, 10 epoch, seed 0)

### 오늘 확인한 shape와 수식

- chain rule 정리: 한 경로 위에서는 국소 미분을 곱하고, 같은 값이 여러 군데 쓰이면 경로끼리 더한다. 행렬곱 `@`가 "곱하고 더하기"를 한 번에 한다. 국소 미분 행렬을 통째로 만들지 않고 dout을 들고 뒤에서부터 오는 이유는 그 행렬이 너무 크고(B = 128이면 Affine 1의 ∂y/∂x만 (128·128) × (128·784) ≈ 16억 칸) 대부분 0이라서다
- SGD: `param ← param − lr · grad`. grad는 "파라미터가 조금 커지면 L이 얼마나 커지나"라서 반대로 간다. dW[0, 0] = 7이면 W[0, 0]을 줄여야 L이 줄어든다
- 제자리 갱신: `Affine(params["W1"], ...)`로 만들었기 때문에 `params["W1"]`과 `affine1.W`는 같은 배열이다. `a = a - 1`은 새 배열을 만들어 이름표만 옮기고, `a -= 1`은 원래 배열의 내용을 바꾼다
- 입력/출력 shape (baseline 한 배치, B = 128, H = 128):
  - `x (128, 784)` → logits `(128, 10)` → loss 스칼라 → dlogits `(128, 10)` → grads는 각 params와 같은 shape
  - 60000 % 128 = 96이라 epoch마다 배치가 469개, 마지막 배치는 96개
  - 파라미터 수: 784·128 + 128 + 128·10 + 10 = 101,770
- 손으로 확인한 계산:
  - SGD: W = 1.0, grad = 1.0, lr = 0.1 → 0.9. `+=` 오른쪽에 param을 다시 넣으면 1 + (1 − 0.1) = 1.9가 된다
  - 첫 epoch loss 0.4561은 초기 loss가 아니다. epoch loss는 469개 배치 loss의 평균이고, 그동안에도 update가 계속 돈다. 학습 전 MNIST 128개의 loss는 2.4359, lr = 0 smoke의 첫 epoch loss는 2.3819 (ln 10 = 2.3026 근처)
  - 학습 전 test 정확도 9.87%: 초기 모델은 10000개 중 7587개를 2로 예측했다. 찍기(1/10) 수준
- MNIST baseline: loss 0.4561 → 0.0903, test 정확도 9.87% → 97.05% (train 97.72%), 약 2초
- 반드시 성립해야 하는 조건:
  - update 뒤에도 `params[key]`는 원래 배열과 같은 객체여야 한다 (`is`)
  - lr = 0이면 loss는 ln 10 근처에서 움직이지 않는다
  - 순서: forward가 저장한 `x`, `mask`, `y_prob`이 있어야 backward가 돌고, backward가 `model.grads`를 채워야 update가 그 값을 쓴다
  - seed가 같으면(초기 W, 배치 섞는 순서) 결과가 소수점까지 같다. baseline을 두 번 돌려 같은 값을 확인했다

### 막힌 지점과 해결 과정

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| params 키 개수를 모르는데 전부 갱신해야 함 | 키를 하나하나 적으려고 했다 | dict를 for문으로 돌면 키가 다 나온다. params와 grads는 키가 같아서 key 하나로 둘 다 꺼낸다 |
| `TypeError: TwoLayerNet.backward() missing 1 required positional argument: 'dout'` | `criterion.backward()`를 호출만 하고 반환값(∂L/∂logits)을 버렸다. `optimizer.update()`에도 인자를 안 줬다 | 반환값을 `model.backward()`에 바로 넘기고, update에는 `model.params`, `model.grads`를 넘긴다 |

### 참고 기록

| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| AI (Claude Code) | chain rule(경로 곱과 경로 합) | `test_sgd.py` 4개, `test_training_smoke.py` 6개 통과, `selfcheck --all` 36 passed, 같은 seed로 baseline을 두 번 돌려 같은 값 |
| 과제 docstring·테스트 | SGD 제자리 갱신 경고(새 배열 대입 금지), train TODO 1~4 힌트, smoke 테스트의 lr = 0 초기 loss 검사 | W = 1, grad = 1, lr = 0.1 손계산과 테스트 기대값 비교 |
