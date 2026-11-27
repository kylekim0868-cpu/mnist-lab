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